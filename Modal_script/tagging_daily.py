import os
import modal

LOCAL = False
DAYS = 3
N_SOUNDS = 5
if LOCAL:
    os.environ["FS_KEY"] = ""

if LOCAL is False:
    stub = modal.Stub("freesound_inference")
    my_image = modal.Image.debian_slim().pip_install(["hopsworks", "requests", "librosa", "numpy", "scikit-learn", "joblib", "torch", "matplotlib", "resampy"])

    @stub.function(image=my_image, schedule=modal.Period(days=DAYS), secrets=[modal.Secret.from_name("my-custom-secret"),
                   modal.Secret.from_name("my-custom-secret-2")])
    def f():
        g()


def g():
    import hopsworks
    import requests
    import librosa
    import numpy as np
    import datetime
    import urllib
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    import torch
    import json
    import matplotlib.pyplot as plt

    project = hopsworks.login()

    model_dir = "best_model"
    if not LOCAL:
        model_registry = project.get_model_registry()
        model_p = model_registry.get_model("freesound_model", version=4)
        model_dir = model_p.download()
    model_path = model_dir + "/entire_model"
    scaler_path = model_dir + "/scaler_top_1000.pkl"
    id2labels_path = model_dir + "/id2token_top_1000.json"
    py_file_path = model_dir + "/CNN_model.py"
    os.replace(py_file_path, "CNN_model.py")

    resource_path = "Resources/Freesound/"
    json_file_path = resource_path + "prd_lbl_tokens.json"
    if not os.path.exists(resource_path):
        os.makedirs(resource_path)

    dataset_api = project.get_dataset_api()
    dataset_api.download(json_file_path, resource_path, overwrite=True)
    with open(json_file_path, "r") as jsf:
        prd_lbl_tokens_dict = json.load(jsf)

    ids, uris = get_all_id_uri()
    n_sounds = min(N_SOUNDS, len(ids))
    ids = ids[0:n_sounds]  # they are sorted by most recent, not by idx order
    uris = uris[0:n_sounds]
    ids = [int(id) for id in ids]
    ids, uris = zip(*sorted(zip(ids, uris)))  # i want them idx order
    download_sound_files(resource_path, ids, uris)

    scaler = load_scaler(scaler_path)
    model = torch.load(model_path)
    model.eval()
    for i in range(0, n_sounds):
        s_file_name = resource_path + f'sound_{i}.ogg'
        img_file_name = resource_path + f'img_{i}.png'
        x = get_x(s_file_name, scaler)
        save_fig(x, img_file_name, ids[i])
        x = torch.from_numpy(x)
        x = torch.unsqueeze(x, 0)
        out = model.forward(x)
        prd_labels = out2labels(id2labels_path, out)
        prd_lbl_tokens_dict[str(ids[i])] = prd_labels

        dataset_api.upload(img_file_name, resource_path, overwrite=True)
        dataset_api.upload(s_file_name, resource_path, overwrite=True)

    prd_lbl_tokens_json = json.dumps(prd_lbl_tokens_dict)
    with open(json_file_path, "w") as of:
        of.write(prd_lbl_tokens_json)
    dataset_api.upload(json_file_path, resource_path, overwrite=True)


def add_key(url):
    return url + f'&token={os.environ["FS_KEY"]}'


def get_all_id_uri():
    import datetime
    import urllib
    query = ""
    yesterday_dat = datetime.datetime.now() - datetime.timedelta(days=DAYS)
    yesterday_dat = yesterday_dat.isoformat()
    filter_dur = "duration:[1 TO 10]"
    filter_dat = f"created:[{yesterday_dat} TO NOW]"
    filter = urllib.parse.quote(filter_dur + " " + filter_dat)
    fields = "id,previews"
    args = f"query={query}&filter={filter}&fields={fields}"
    url = f'https://freesound.org/apiv2/search/text/?{args}'

    ids = []
    uris = []
    while url:
        url = add_key(url)
        ids, uris, url = get_page_id_uri(url, ids, uris)

    return ids, uris


def get_page_id_uri(url, ids, uris):
    import requests
    r = requests.get(url)
    rj = r.json()
    next_url = rj['next']
    results = rj['results']
    page_counts = len(results)

    for i in range(0, page_counts):
        ids.append(results[i]['id'])
        uris.append(results[i]['previews']['preview-hq-ogg'])

    return ids, uris, next_url


def download_sound_files(resource_path, ids, uris):
    import requests
    with requests.Session() as sesh:
        for i in range(0, len(uris)):
            download = sesh.get(uris[i])
            if download.status_code == 200:
                with open(resource_path + f"sound_{i}.ogg", 'wb') as f:
                    f.write(download.content)
            else:
                print(f'download failed file {id[i]}')


def load_scaler(file):
    import joblib
    from sklearn.preprocessing import MinMaxScaler
    with open(file, 'rb') as f:
        scaler = joblib.load(f)
        scaler.clip = True
    return scaler


def pad(l, sr):
    import numpy as np
    z = np.zeros(10*sr, dtype='float32')
    z[:l.size] = l
    z = z + 5*1e-4*np.random.rand(z.size).astype('float32')
    return z


def compute_log_mel_spec(filename, sr=22000, n_mels=96):
    import librosa
    audio, sr = librosa.load(filename, sr=sr, res_type='kaiser_fast')
    x = pad(audio, sr=sr)
    y_log_mel = librosa.power_to_db(librosa.feature.melspectrogram(y=x, sr=sr, hop_length=512, n_fft=1024, n_mels=96))
    return y_log_mel


def return_spectrogram_max_nrg_frame(spectrogram):
    import librosa
    import numpy as np
    frames = librosa.util.frame(np.asfortranarray(spectrogram), frame_length=96, hop_length=12)
    idx_max_nrg = np.argmax(np.sum(np.sum(frames, axis=0), axis=0))
    return frames[:, :, idx_max_nrg]


def get_x(path, scaler):
    import librosa
    from sklearn.preprocessing import MinMaxScaler
    mel_spec = compute_log_mel_spec(filename=path)
    x = return_spectrogram_max_nrg_frame(mel_spec)
    x = scaler.transform(x)
    return x


def out2labels(path, out):
    import torch
    import json
    k = 5
    out = out.detach().cpu()
    top_log, top_ind = torch.topk(out, k)
    top_ind = top_ind.squeeze()
    top_ind = top_ind.numpy()
    with open(path) as jsf:
        id2labels = json.load(jsf)

    labels = []
    for ind in top_ind:
        labels.append(id2labels[str(ind)])

    return labels


def save_fig(x, img_path, id):
    import matplotlib.pyplot as plt
    plt.imshow(x, cmap="viridis")
    plt.title(id)
    plt.savefig(img_path)


if __name__ == "__main__":
    if LOCAL is True:
        g()
    else:
        # stub.deploy("freesound_inference")
        with stub.run():
            f()
