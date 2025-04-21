import torchaudio
from scipy.io.wavfile import write
from resemble_enhance.enhancer.inference import denoise, enhance

device = 'cpu'
solver = 'midpoint'
nfe = 64
tau = 0.5

dwav, sr = torchaudio.load('audio.wav')
dwav = dwav.mean(dim=0)

wav1, wav1_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=0.1, tau=tau)

wav1 = wav1.numpy()

write('output.wav', wav1_sr, wav1)