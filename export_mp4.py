import os
from scipy.io.wavfile import write
import torch
from tqdm import tqdm


LRS3_TEST_COUNTS = 1321


def save_wav_16khz(wav_outdir:str, wav:torch.Tensor):
    """saves audio from item in AVHubertDataset

    Args:
        wav_name (str)
        wav (torch.Tensor): of size [T]
    """
    wav = wav.cpu().numpy()
    write(wav_outdir, 16_000, wav)


def generate_evaluation_audio(export_path, gt_path):
    pbar = tqdm(range(LRS3_TEST_COUNTS))
    for root, dirs, files in os.walk(gt_path):
        for file in files:
            rel_path = os.path.relpath(root, gt_path)  # Relative path from path_gt
            if file.endswith('.flac'):
                file = file.replace('.flac', '')
                target_nopostfix = os.path.join(export_path, rel_path, file)
                mp3source = target_nopostfix+'.wav'
                mp4source = os.path.join(gt_path, rel_path, file)+'.mp4'
                command = f"ffmpeg -y -i {mp4source} -i {mp3source} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {target_nopostfix}.mp4>{target_nopostfix}.log 2>&1"
                os.system(command)
                pbar.update()
                pbar.set_description(f"Saved: f{target_nopostfix}.mp4")
            


if __name__ == '__main__':
    eval_path = f"/data1/yfliu/samples/v2s/all_samples/lrs3/diffv2s/test"  # Path where to store evaluated files
    path_gt = "/data1/yfliu/lrs3/test"  # Ground truth path to traverse
    generate_evaluation_audio(eval_path, path_gt)