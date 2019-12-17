# DSP-research

These files are my practice of CNN auto-encoder based on Keras and Librosa library.

My working environment:
Ubuntu 18.04

Data source:
http://speech.ee.ntu.edu.tw/DSP2019Autumn/hw2/dsp_hw2-1.zip
/dsp-hw2-1/speechdata

What is auto-encoder:
We wish the output of the CNN as close as the input in order to fulfill some purpose.

How to execute:
1.Put all five python files and the speechdata folder under the same path.
2.Run train_data_generator.py, which will create train_data.npy file(it stores an 4-dim array).
3.Run test_data_generator.py, which will create test_data.npy file(it also stores an 4-dim array).
4.Run autoencoder_for_many_data.py, which will train a CNN model based on training_data under speechdata folder, and store it in my_model_for_many_data.h5 file.
5.Run evaluate_for_many_data.py, which will evaluate how the CNN model performs based on testing_data under speechdata folder.
6.Modify predict_for_one_data.py to load any audio file that you want to feed into CNN model.(This model is only able to work with the audio file length around 1 second)
7.Run predict_for_one_data.py, and you will see the waveform of the original audio file and the waveform of the output of CNN. Also, it will create reconstruct.wav that is exactly the final output.
8.You can check whether reconstruct.wav sounds similar to original one.
Acknowledgement:The result turns out not as well as expected. The reconstruct audio sounds really ambiguous and noisy. It is hardly to say that one can know the original content through the reconstrctive counterpart. I believe the reason is that I didn't train the model well.

How it works:
1.Convert training audio to log-mel spectrogram with Librosa library.(spectrogram = 2-dim array)
2.Normalize the array and reshape it into 3-dim array.((128x48)->(128x48x1))
3.For each audio file we can get corresponding 3-dim array, stacking them to create a 4-dim array.
4.Use them to train CNN.
5.Next, convert testing audio to 4-dim array as previous steps.
6.Feed them into CNN to evaluate the performance.
7.Finally, you can try to feed any audio around 1 second length.
8.CNN will output a 4-dim array.
9.The code will reshape it into a 2-dim array(spectrogram), and then denormalize it and convert it back to an audio clip.
