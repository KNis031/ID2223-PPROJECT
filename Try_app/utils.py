# !!! NOTE: these may be slightly different from functions with the same name
# at other locations


def pad(l, sr):
    import numpy as np
    # 0-Pad 10 sec at fs hz and add little noise
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
    import numpy as np
    import librosa
    frames = librosa.util.frame(np.asfortranarray(spectrogram), frame_length=96, hop_length=12)
    idx_max_nrg = np.argmax(np.sum(np.sum(frames, axis=0), axis=0))
    return frames[:, :, idx_max_nrg]


def load_scaler(file):
    import joblib
    from sklearn.preprocessing import MinMaxScaler
    with open(file, 'rb') as f:
        scaler = joblib.load(f)
        scaler.clip = True
    return scaler


def add_key(url):
    return url + f'&token={KEY}'


def get_all_id_uri():
    import datetime
    import urllib
    query = ""
    yesterday_dat = datetime.datetime.now() - datetime.timedelta(days=7)
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
    # print(r.text)
    next_url = rj['next']
    results = rj['results']
    page_counts = len(results)

    for i in range(0, page_counts):
        ids.append(results[i]['id'])
        uris.append(results[i]['previews']['preview-hq-ogg'])

    return ids, uris, next_url


def download_files(ids, uris):
    import requests
    with requests.Session() as sesh:
        for i in range(0, len(uris)):
            download = sesh.get(uris[i])
            if download.status_code == 200:
                name = str(ids[i]) + '.ogg'
                with open('sounds/' + name, 'wb') as f:
                    f.write(download.content)
            else:
                print(f'download failed file {id[i]}')


def dowload_sounds():
    ids, uris = get_all_id_uri()
    download_files(ids=ids, uris=uris)


def get_x(path, scaler):
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
    ax = plt.imshow(x, cmap="viridis")
    plt.title(id)
    plt.savefig(img_path)
    return ax


if __name__ == '__main__':
    import torch
    # dowload_sounds()
    scaler = load_scaler('best_model/scaler_top_1000.pkl')
    dir = 'sounds/'
    file_name = '717730.ogg'
    sound_path = dir + file_name

    x = get_x(sound_path, scaler)
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 0)
    print(x.size())
    model = torch.load('best_model/entire_model')
    model.eval()
    out = model.forward(x)
    out = out.detach().cpu()
    prd_labels = out2labels(out)
    print(prd_labels)