from meta_relavant_files.traj_vae import VaeDecoder, TrajVAE
import torch 

# save
def save_model(model):
    model_PATH = " "
    torch.save({
        'encoder_dict': model.vae_encoder.state_dict(),
        'decoder_dict': model.vae_decoder.state_dict(),
        'traj_vae_dict': model.state_dict(),
    })



def load_model():
    checkpoint = torch.load(/home/SENSETIME/zhoutong/hoffnung/xad/result/vae_decoder_ckpt)
    modelA = VAEDecoder()
    modelB = TrajVAE()
    modelA.load_state_dict(checkpoint['decoder_dict'])
    modelB.load_state_dict(checkpoipnt['traj_vae_dict'])
    modelA.eval()
    modelB.eval()
    # or
    modelA.train()
    modelA.train()


######## Examples #############################################
Save:
torch.save(model.state_dict(), PATH)

Load:
modelB = TheModelClass*(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)
modelB.eval()

