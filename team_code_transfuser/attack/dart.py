from attackmethod import Pixle, AutoAttack, VMIFGSM, CW
from RSMamba import RSM_SS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

class Domain_Discriminator(nn.Module):
    def __init__(
        self,
        feature_dims,        
        kernel_size,
        stride,
        padding,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.discriminator = nn.Sequential(
            nn.Conv2d(feature_dims, feature_dims // 2, kernel_size, stride, padding),
            nn.BatchNorm2d(feature_dims//2),
            nn.ReLU(),
            nn.Conv2d(feature_dims//2, 32, kernel_size, stride, padding)
        )
        self.flatten = nn.Sequential(
            nn.AdaptivePool2d((1,1)),
            nn.Flatten(),        
        )

    def forward(self,x):
        x = self.discriminator(x)
        x = self.flatten(x)
        return x

class PseudoLabelling(nn.Module):
    def __init__(
        self,
        checkpoint,
        model = RSM_SS(), # we have use classifier here
        device = 'cpu',         
    ):
        super().__init__()
        self.device = device

        # initialize the the pseudo-label predictor with the same architecture as the model
        self.hp = model
        self.hp.to(device)

        # set eval stage
        self.hp.eval()

        # set grads stage
        for param in self.hp.parameters():
            param.require_grads = False
        for param in self.hp.parameters():
            param.require_grads = False

    def forward(self,FE_result): 
        # feature extractor
        FE_result = FE_result.to(self.device)
        with torch.no_grad():
            pseudo_labels = self.hp(FE_result)        
        return pseudo_labels


class DART:
    def __init__(self, feature_extractor, classifier, domain_discriminator, source_loader, target_loader, device, pseudo_label_predictor = PseudoLabelling()):
        assert isinstance(feature_extractor, nn.Module), "feature_extractor must be an instance of nn.Module"
        assert isinstance(classifier, nn.Module), "classifier must be an instance of nn.Module"
        assert isinstance(domain_discriminator, nn.Module), "domain_discriminator must be an instance of nn.Module"
        assert isinstance(source_loader, DataLoader), "source_loader must be an instance of DataLoader"
        assert isinstance(target_loader, DataLoader), "target_loader must be an instance of DataLoader"
        assert device in ['cpu', 'cuda'], "device must be either 'cpu' or 'cuda'"
        
        if pseudo_label_predictor is None:
            pseudo_label_predictor = PseudoLabelling()
        assert isinstance(pseudo_label_predictor, nn.Module), "pseudo_label_predictor must be an instance of nn.Module"
            
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.domain_discriminator = domain_discriminator
        # we need to work with this, let create a full loading data
        self.source_loader = source_loader # main source training with label
        self.target_loader = target_loader # unspervised non-labelled target 
        self.device = device
        self.pseudo_label_predictor = pseudo_label_predictor # normally we will use same predictor from model that we have
        self.criterion = nn.CrossEntropyLoss()
        
        self.update_pseudo_labels_freq = 10
        self.checkpoint_freq = 20

        self.writer = SummaryWriter()
    def train_epoch(self, epoch, optimizer, lambda_1, lambda_2, lr,lr_d, weight_decay, weight_decay_d):
            self.feature_extractor.train()
            self.classifier.train()
            self.domain_discriminator.train()

            for (source_data, source_labels), (target_data, _) in zip(self.source_loader, self.target_loader):
                source_data, source_labels = source_data.to(self.device), source_labels.to(self.device)
                target_data = target_data.to(self.device)

                # Generate adversarially perturbed source data
                transformed_source_data = self.generate_adversarial_examples(source_data, source_labels)

                # Generate pseudo-labels for target data using the pseudo-label predictor
                with torch.no_grad():
                    pseudo_labels = self.pseudo_label_predictor(target_data).argmax(dim=1)

                # Optimize feature extractor and classifier
                optimizer.zero_grad()
                source_features = self.feature_extractor(transformed_source_data)
                source_preds = self.classifier(source_features)
                source_loss = self.criterion(source_preds, source_labels)

                target_features = self.feature_extractor(target_data)
                target_preds = self.classifier(target_features)
                target_loss = self.criterion(target_preds, pseudo_labels)

                # Generate adversarially perturbed target data
                perturbed_target_data = self.generate_adversarial_examples(target_data, pseudo_labels)
                perturbed_target_features = self.feature_extractor(perturbed_target_data)

                domain_loss = lambda_1 * self.domain_divergence(source_features, perturbed_target_features)

                # Compute the ideal joint loss
                joint_loss = self.ideal_joint_loss(source_data, source_labels, perturbed_target_data, pseudo_labels)

                total_loss = source_loss + lambda_2 * target_loss + domain_loss + joint_loss
                total_loss.backward()
                self.optimize_model(source_features, perturbed_target_features, total_loss, lr,lr_d, weight_decay, weight_decay_d)

                self.writer.add_scalar('Train/source_loss', source_loss.item(), epoch)
                self.writer.add_scalar('Train/target_loss', target_loss.item(), epoch)
                self.writer.add_scalar('Train/domain_loss', domain_loss.item(), epoch)
    
    def optimize_model(self,source_features, perturbed_target_features, total_loss, lr,lr_d, weight_decay, weight_decay_d):
        optimizer = optim.Adam(list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
                               lr=lr, weight_decay=weight_decay)
        optimizer_d = optim.Adam(self.domain_discriminator.parameters(), lr=lr_d, weight_decay=weight_decay_d)

        # Optimize feature extractor and classifier
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Optimize domain discriminator
        optimizer_d.zero_grad()
        domain_loss_d = self.domain_divergence(source_features.detach(), perturbed_target_features.detach())
        domain_loss_d.backward()
        optimizer_d.step()

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')

    def evaluate(self, test_loader, best_model_path):
        # Load the best model checkpoint
        checkpoint = torch.load(best_model_path)
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])

        self.feature_extractor.eval()
        self.classifier.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                # Extract features
                features = self.feature_extractor(data)

                # Classify the data
                preds = self.classifier(features)
                loss = self.criterion(preds, labels)

                total_loss += loss.item()

                # Compute accuracy
                _, predicted_labels = torch.max(preds, 1)
                correct_predictions += (predicted_labels == labels).sum().item()
                total_predictions += labels.size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / total_predictions

        # Log the test metrics
        self.writer.add_scalar('Test/Loss', avg_loss)
        self.writer.add_scalar('Test/Accuracy', accuracy)

        # Save the classifier's state dictionary
        torch.save(self.classifier.state_dict(), 'best_classifier.pt')

        return avg_loss, accuracy
    def validate(self, validation_loader, epoch):
        self.feature_extractor.eval()
        self.classifier.eval()
        self.domain_discriminator.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for (source_data, source_labels), (target_data, _) in zip(validation_loader['source'], validation_loader['target']):
                # ... (code for validation step remains the same)
                source_data, source_labels = source_data.to(self.device), source_labels.to(self.device)
                target_data = target_data.to(self.device)

                # Extract features
                source_features = self.feature_extractor(source_data)
                target_features = self.feature_extractor(target_data)

                # Classify source data
                source_preds = self.classifier(source_features)
                source_loss = self.criterion(source_preds, source_labels)

                # Generate pseudo-labels for target data using the pseudo-label predictor
                pseudo_labels = self.pseudo_label_predictor(target_features).argmax(dim=1)

                # Classify target data
                target_preds = self.classifier(target_features)
                target_loss = self.criterion(target_preds, pseudo_labels)

                # Compute domain divergence
                domain_loss = self.domain_divergence(source_features, target_features)

                # Compute the ideal joint loss
                joint_loss = self.ideal_joint_loss(source_data, source_labels, target_data, pseudo_labels)

                # Compute total loss
                loss = source_loss + self.lambda_2 * target_loss + self.lambda_1 * domain_loss + joint_loss
                total_loss += loss.item()

                # Compute accuracy
                _, predicted_labels = torch.max(source_preds, 1)
                correct_predictions += (predicted_labels == source_labels).sum().item()
                total_predictions += source_labels.size(0)

        avg_loss = total_loss / len(validation_loader['source'])
        accuracy = correct_predictions / total_predictions

        # Log the validation metrics
        self.writer.add_scalar('Validation/Loss', avg_loss, epoch)
        self.writer.add_scalar('Validation/Accuracy', accuracy, epoch)

        return avg_loss, accuracy

    def train(self, epochs, lr, lr_d, weight_decay, weight_decay_d, lambda_1, lambda_2,validation_loader):
        

        for epoch in range(epochs):
            self.train_epoch(epoch, lr, lr_d, weight_decay, weight_decay_d, lambda_1, lambda_2)     
        
            if (epoch + 1) % self.update_pseudo_labels_freq == 0:
                self.update_pseudo_label_predictor(validation_loader)
            
            if (epoch + 1) % self.checkpoint_freq == 0:
                self.save_checkpoint(epoch)

    def generate_adversarial_examples(self,attack_type, images, labels):
        # TODO: Implement adversarial example generation (e.g., PGD, FGSM) # pleaes do PGD as following the paper
        # Return the perturbed data
        # Pixle, AutoAttack, VMIFGSM, CW
        if attack_type== "Pixle":
            attack = Pixle(self.classifier, x_dimensions = (0.1, 0.2), restarts = 10, max_iterations = 50)
            adv_images = attack(images, labels)
        elif attack_type == "AutoAttack":
            attack = AutoAttack(self.classifier, norm = 'Linf', eps = 8/255, version = 'standard', n_classes = 10, seed = None, verbose = False)
            adv_images = attack(images, labels)
        elif attack_type == "VMIFGSM":
            attack = VMIFGSM(self.classifier, eps = 8/255, alpha = 2/255, steps = 10, decay = 1.0, N = 5, beta = 3/2)
            adv_images = attack(images, labels)
        elif attack_type == "CW":
            attack = CW(self.classifier, c = 1, kappa = 0, steps = 50, lr = 0.01)
            adv_images = attack(images, labels)        
        else:
            return ValueError("Only Pixle, AutoAttack, VMIFGSM, CW are available ")
        return adv_images
            

    def domain_divergence(self, divergence_type, source_features, target_features, domain_discriminator):
        # implement DANN, MMD,CMD, CORAL,KL, WASSERSTEIN, TRADEs
        # need to clarify what is source, target features, is it a product of feature_extractor
        if divergence_type == "DANN":
            return self.dann_divergence(source_features, target_features, domain_discriminator)
        elif divergence_type == "MMD":
            return self.mmd_divergence(source_features, target_features, kernel_type = 'rbf', kernel_mul = 2.0, kernel_num = 5)
        elif divergence_type == "CMD":
            return self.cmd_divergence(source_features, target_features, moments = [2,3])
        elif divergence_type == "Coral":
            return self.coral_divergence(source_features, target_features)
        elif divergence_type == "kl":
            return self.kl_divergence(source_features, target_features, temperature = 1.0)
        elif divergence_type == "wass":
            return self.wasserstein_divergence(source_features, target_features, domain_discriminator, lambda_gp=10)
        else:
            return ValueError(" Only DANN, MMD,CMD, CORAL,KL, WASSERSTEIN method are available")
    
    # i think ideal joint loss have problem
    def ideal_joint_loss(self, source_data, source_labels, target_data, pseudo_labels):
        # Compute the minimum loss of the classifier on the source domain
        source_features = self.feature_extractor(source_data)
        source_preds = self.classifier(source_features)
        source_loss = self.criterion(source_preds, source_labels)

        # Generate adversarial examples for the target data
        perturbed_target_data = self.generate_adversarial_examples(target_data, pseudo_labels)
        
        # Compute the minimum loss of the classifier on the worst-case target domain
        perturbed_target_features = self.feature_extractor(perturbed_target_data)
        perturbed_target_preds = self.classifier(perturbed_target_features)
        target_loss = self.criterion(perturbed_target_preds, pseudo_labels)

        # Compute the ideal joint loss
        joint_loss = source_loss + target_loss

        return joint_loss
    def update_pseudo_label_predictor(self, validation_loader):
        self.feature_extractor.eval()
        self.classifier.eval()

        # Evaluate the model's performance on the validation set
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in validation_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                features = self.feature_extractor(data)
                outputs = self.classifier(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        # Update the pseudo-label predictor if the model's performance improves
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.pseudo_label_predictor.load_state_dict(self.classifier.state_dict())

    def dann_divergence(source_features, target_features, domain_discriminator):
        source_domain_labels = torch.zeros(source_features.size(0)).to(source_features.device)
        target_domain_labels = torch.ones(target_features.size(0)).to(target_features.device)
        
        domain_preds_source = domain_discriminator(source_features)
        domain_preds_target = domain_discriminator(target_features)
        
        domain_loss = nn.BCEWithLogitsLoss()(domain_preds_source, source_domain_labels) + \
                    nn.BCEWithLogitsLoss()(domain_preds_target, target_domain_labels)
        
        return domain_loss

    def mmd_divergence(source_features, target_features, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        n_source , n_target = source_features.size(0), target_features.size(0)
        kernels = []
        for i in range(kernel_num):
            alpha = kernel_mul ** i
            if kernel_type == 'rbf':
                bandwidth = torch.mean(torch.cdist(source_features,target_features))
                kernel = torch.exp(-alpha * (torch.cdist(source_features, target_features) ** 2) / (2 * bandwidth ** 2))
            elif kernel_type == 'linear':
                kernel = alpha * torch.matmul(source_features, target_features.t())
            else:
                raise ValueError(f"Unsupported kernel type: {kernel_type}")
            kernels.append(kernel)
        
        kernel_sum = sum(kernels)
        mmd_loss = (kernel_sum / (n_source * n_source)).sum() - 2 * (kernel_sum / (n_source * n_target)).sum() + (kernel_sum / (n_target * n_target)).sum()
            
        return mmd_loss
    def cmd_divergence(source_features, target_features, moments=[2, 3]):
        loss = 0
        for k in moments:
            source_moment = torch.mean(source_features ** k, dim=0)
            target_moment = torch.mean(target_features ** k, dim=0)
            loss += torch.norm(source_moment - target_moment, p=2)
        
        return loss

    def coral_divergence(source_features, target_features):
        source_features_centered = source_features - torch.mean(source_features, dim=0)
        target_features_centered = target_features - torch.mean(target_features, dim=0)
        
        source_cov = torch.matmul(source_features_centered.t(), source_features_centered) / (source_features.size(0) - 1)
        target_cov = torch.matmul(target_features_centered.t(), target_features_centered) / (target_features.size(0) - 1)
        
        coral_loss = torch.norm(source_cov - target_cov, p='fro') ** 2
        
        return coral_loss

    def kl_divergence(source_features, target_features, temperature=1.0):
        source_softmax = nn.Softmax(dim=1)(source_features / temperature)
        target_softmax = nn.Softmax(dim=1)(target_features / temperature)
        
        kl_loss = nn.KLDivLoss(reduction='batchmean')(torch.log(source_softmax), target_softmax)
        
        return kl_loss

    def wasserstein_divergence(source_features, target_features, discriminator, lambda_gp=10):
        source_preds = discriminator(source_features)
        target_preds = discriminator(target_features)
        
        wd_loss = torch.mean(source_preds) - torch.mean(target_preds)
        
        # Gradient penalty
        alpha = torch.rand(source_features.size(0), 1).to(source_features.device)
        interpolates = alpha * source_features + (1 - alpha) * target_features
        interpolates.requires_grad_(True)
        disc_interpolates = discriminator(interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(disc_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
        
        wd_loss += gradient_penalty
        
        return wd_loss

