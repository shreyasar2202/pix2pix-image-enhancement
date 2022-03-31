import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image 
from IPython.display import display

def normalize(img):
    return (img-np.min(img))/(np.max(img)-np.min(img))
   
def globalToneMapping(image, Lwhite, alpha):
    """Global tone mapping using gamma correction
    ----------
    images : <numpy.ndarray>
        Image needed to be corrected
    Lwhite : floating number
        The number for constraint the highest value in hdr image
    alpha : floating number
        The number for correction. Higher value for brighter result; lower for darker
    Returns
    -------
    numpy.ndarray
        The resulting image after gamma correction
    """
    delta = 1e-9
    N = image.shape[0]*image.shape[1]*3
    #Lw = np.exp(image)
    Lw = image
    Lb = np.exp((1/N)*np.sum(np.log(delta+Lw)))
    Lm = (alpha/Lb)*Lw # linear 
    Ldf = Lm * (1 + (Lm / (Lwhite**2))) / (1 + Lm)
    image_corrected = Ldf
    image_corrected = ((image_corrected - image_corrected.min())*(255/(image_corrected.max()-image_corrected.min())))
    return image_corrected

def visualize(ldr_,hdr_,generated_,epoch, figsize=(10,10)):
    
    plt.clf()
    ldr = ldr_.copy()
    hdr = hdr_.copy()
    generated = generated_.copy()
    #print(generated.max(),generated.min())
    hdr = globalToneMapping(hdr, Lwhite=np.exp(hdr.max())*0.8, alpha=0.4).astype(np.uint8)
    generated = globalToneMapping(generated, Lwhite=np.exp(generated.max())*0.8, alpha=0.4).astype(np.uint8)
    #print(generated.max(),generated.min())
    
    fig, axs = plt.subplots(1, 3, figsize=(15,15))
    axs[0].imshow(normalize(ldr))
    axs[0].set_title('LDR')
    axs[1].imshow(hdr)
    axs[1].set_title('HDR')
    axs[2].imshow(generated)
    axs[2].set_title('Generated')
    plt.show()
    #plt.savefig('experiment_data/plots/epoch'+str(epoch)+'.png')
    
def visualize_test(ldr_,hdr_,generated_, figsize=(10,10)):
    
    plt.clf()
    ldr = ldr_.copy()
    hdr = hdr_.copy()
    generated = generated_.copy()
    #print(generated.max(),generated.min())
    hdr = globalToneMapping(hdr, Lwhite=np.exp(hdr.max())*0.8, alpha=0.4).astype(np.uint8)
    generated = globalToneMapping(generated, Lwhite=np.exp(generated.max())*0.8, alpha=0.4).astype(np.uint8)
    #print(generated.max(),generated.min())
    
    fig, axs = plt.subplots(1, 3, figsize=(15,15))
    axs[0].imshow(normalize(ldr))
    axs[0].set_title('LDR')
    axs[1].imshow(hdr)
    axs[1].set_title('HDR')
    axs[2].imshow(generated)
    axs[2].set_title('Generated')
    plt.show()
    #plt.savefig('experiment_data/plots/epoch'+str(epoch)+'.png')
    
    
