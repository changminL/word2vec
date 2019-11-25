import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

torch.manual_seed(2019)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024, 28*28)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        z = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        x = x.view(z.shape[0],1,28,28)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28,512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, img):
        img_flat = img.view(img.shape[0],-1)
        v = self.fc1(img_flat)
        v = self.relu(v)
        v = self.fc2(v)
        v = self.relu(v)
        v = self.fc3(v)
        v = self.sigmoid(v)
        return v

def load_data(batch_size):
    train_dataset = torchvision.datasets.MNIST(root ='../data',
                                               train = True,
                                               transform = transforms.ToTensor(),
                                               download = True)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = batch_size,
                                               shuffle = True)
    return train_dataset, train_loader

def data_visualize(train_loader):
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    figure = plt.figure()
    num_images = 60
    for idx in range(0, num_images):
        plt.subplot(6, 10, idx+1)
        plt.axis('off')
        plt.imshow(images[idx].numpy().squeeze(), cmap = 'gray_r')
    plt.show()

def train(device, dataloader, generator, discriminator, optimizer_G, optimizer_D):
    generator.train()
    discriminator.train()
    num_epochs = 100
    k = 2
    CE_loss = nn.BCELoss()
    generator.to(device)
    discriminator.to(device)
    G_loss_per_epoch = []
    D_loss_per_epoch = []
    epochs = []
    for e in range(num_epochs):
        D_avg_loss = 0
        G_avg_loss = 0
        G_count = 0
        D_count = 0
        for i, (imgs,_) in enumerate(dataloader):

            imgs = imgs.to(device)
            valid = torch.ones((imgs.shape[0], 1),dtype=torch.float32).to(device)
            fake = torch.zeros((imgs.shape[0], 1),dtype=torch.float32).to(device)
            real_imgs = imgs.clone().detach()
            x = torch.normal(mean=0,std=1,size=(imgs.shape[0],latent_dim)).to(device)
            gen_imgs = generator(x)

            if i % k == 0:
                optimizer_G.zero_grad()
                g_loss = CE_loss(discriminator(gen_imgs),valid)
                G_avg_loss += g_loss
                G_count += 1
                g_loss.backward()
                optimizer_G.step()
                if i % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], G_loss {:.4f}'.format(e, num_epochs, i, len(dataloader), g_loss))

            optimizer_D.zero_grad()
            real_loss = CE_loss(discriminator(real_imgs),valid)
            fake_loss = CE_loss(discriminator(gen_imgs.detach()),fake)
            d_loss = real_loss + fake_loss
            D_avg_loss += d_loss
            D_count += 1
            d_loss.backward()
            optimizer_D.step()
            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], D_loss {:.4f}'.format(e, num_epochs, i,len(dataloader), d_loss))
        D_avg_loss = D_avg_loss / float(D_count)
        G_avg_loss = G_avg_loss / float(G_count)
        G_loss_per_epoch.append(G_avg_loss)
        D_loss_per_epoch.append(D_avg_loss)
        epochs.append(e)
        if e == 10 or e == 30 or e == 50 or e == 70 or e == 90:
            test_generator(25, 100)
    return G_loss_per_epoch, D_loss_per_epoch, epochs

def test_generator(batch_size, latent_dim):
    x = torch.normal(mean=0, std=1, size=(batch_size, latent_dim))
    generator = Generator(latent_dim)
    gen_imgs = generator(x)
    figure = plt.figure()
    num_images = batch_size
    for idx in range(1, num_images + 1):
        plt.subplot(5,5,idx)
        plt.axis('off')
        plt.imshow(gen_imgs[idx-1].detach().numpy().squeeze(), cmap='gray_r')
    plt.show()

if __name__ == '__main__':
    batch_size = 64
    latent_dim = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, train_loader = load_data(batch_size)

    #test_generator(batch_size, latent_dim)
    #data_visualize(train_loader)
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    generator = Generator(latent_dim)
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(),lr=lr,betas=(b1,b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=lr,betas=(b1,b2))

    g_loss, d_loss, epochs = train(device, train_loader, generator, discriminator, optimizer_G, optimizer_D)
    '''
    my_model = Net(28*28, 500, 10).to(device)
    learning_rate = 0.001
    betas = (0.9, 0.999)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate, betas=betas)

    num_epochs = 10
    total_step = len(train_loader)
    '''
    '''
    x_axis = []
    y_axis = []
    for epoch in range(num_epochs):
        avg_loss = train(my_model, device, train_loader, optimizer, epoch, num_epochs, criterion, total_step)
        x_axis.append(epoch)
        y_axis.append(avg_loss)
    '''
    plt.plot(epochs, g_loss)
    plt.show()
    plt.plot(epochs, d_loss)
    plt.show()
    #test(my_model, device, test_loader)
    #torch.save(my_model.state_dict(), 'model.ckpt')

