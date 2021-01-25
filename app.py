import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from datetime import time
import datetime
import plotly.graph_objects as go
import plotly.express as px

plt.rcParams["font.size"] = 18
HOP = 1000

@st.cache
def calc_melspectrogram(wav, sr, win_len, hop_len, n_mel):
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=win_len, hop_length=hop_len, win_length=win_len, n_mels=n_mel)
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel

@st.cache
def calc_spectrum(wav, sr):
    spectrum = np.abs(np.fft.fft(wav, sr)[:int(sr/2)])
    freqs = np.fft.fftfreq(sr, d=1.0/sr)[:int(sr/2)]
    s_power = np.abs(spectrum)
    return s_power, freqs

def move_ave(ts,win):
    ts_pad =np.pad(ts,[int(win/2),int(win/2)],'reflect')
    return np.convolve(ts_pad,np.full(win,1/win),mode='same')[int(win/2):-int(win/2)]

def main():

    st.title('audio visualizer')
    uploaded_file = st.sidebar.file_uploader("audio file upload") 

    if uploaded_file is not None:
        wav, sr = librosa.load(uploaded_file, sr=None)
        wav_seconds = int(len(wav)/sr)

        st.write('sampling rate = ', sr, 'Hz')
        st.audio(uploaded_file)

        st.sidebar.title('sound waveform')
        tgt_ranges = st.sidebar.slider("target range(s)", 0, wav_seconds, (0, wav_seconds))
        st.sidebar.title('melspectrogram')
        hop_len = st.sidebar.slider('hop len',  min_value=128, max_value=2048, step=128, value=1024)
        win_len = st.sidebar.slider('win len',  min_value=512, max_value=4096, step=256, value=2048)
        n_mel = st.sidebar.slider('mel num',  min_value=64, max_value=256, step=8, value=128)
        st.sidebar.title('spectrum')
        ave_win_len = st.sidebar.slider('ave win len',  min_value=2, max_value=500, step=2, value=100)

        fig = go.Figure()
        x_wav = np.arange(len(wav)) / sr
        fig.add_trace(go.Scatter(y=wav[::HOP], name="wav"))
        fig.add_vrect(x0=int(tgt_ranges[0]*sr/HOP), x1=int(tgt_ranges[1]*sr/HOP), fillcolor="LightSalmon", opacity=0.5,
                    layer="below", line_width=0)
        fig.update_layout(title="sound waveform", width=800, height=400,
                    xaxis = dict(
                    tickmode = 'array',
                    tickvals = [1, int(len(wav[::HOP])/2), len(wav[::HOP])],
                    ticktext = [str(0), str(int(wav_seconds/2)), str(wav_seconds)],
                    title = "time(s)"
                ))
        st.plotly_chart(fig)

        wav_element = wav[tgt_ranges[0]*sr:tgt_ranges[1]*sr]

        # melspectrogram
        mel = calc_melspectrogram(wav_element, sr, win_len, hop_len, n_mel)
        mel_bins = librosa.mel_frequencies(n_mel, 0, int(sr/2))

        fig = px.imshow(np.flipud(mel), aspect='auto')
        fig.update_layout(title="melspectrogram", width=800, height=400,
                            xaxis = dict(showticklabels=False),
                            yaxis = dict(
                            tickmode = 'array',
                            tickvals = [1, int(mel.shape[0]/4), int(mel.shape[0]/2), int(mel.shape[0]-1)],
                            ticktext = [str(int(mel_bins[int(mel.shape[0]-1)])), str(int(mel_bins[int(3*mel.shape[0]/4)])), str(int(mel_bins[int(mel.shape[0]/2)])), str(0)],
                            title = "frequency(Hz)"))
        st.write(fig)

        # spectrum
        s_power, freqs = calc_spectrum(wav_element, sr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = freqs, y = move_ave(s_power, ave_win_len), mode = 'lines'))
        fig.update_layout(title="spectrum", width=800, height=400, 
                        xaxis = dict(title = "frequency(Hz)"), yaxis = dict(title = "power"))
        st.write(fig)

if __name__ == "__main__":
    main()