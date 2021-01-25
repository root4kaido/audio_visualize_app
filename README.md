# audio_visualize_app

With this application, you can visualize any part of the sound data (melspectrogram, spectrum). \
It has been deployed on [Streamlit Sharing](https://www.streamlit.io/sharing), so anyone can use it from the URL below.

https://share.streamlit.io/root4kaido/audio_visualize_app/main/app.py

![img](https://github.com/root4kaido/audio_visualize_app/blob/main/app.gif)

## How To Use

1. Upload your audio data.

2. From top to bottom, the waveform, melspectrogram, and spectrum are displayed.

3. You can change the parameters from the widget in the sidebar.

4. The top parameter determines the range of the visualization. If you change this parameter, it will change where in the waveform you are focusing (orange area). The merspectrogram and spectrum are calculated using signals from the target range only.

