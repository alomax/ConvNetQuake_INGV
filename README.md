# ConvNetQuake_INGV
## A python package for rapid earthquake characterization using single station waveforms and a convolutional neural network.


## Citation
We have a manuscript submitted on the development and application of ConvNetQuake_INGV, if you reference or make use of ConvNetQuake_INGV please cite the folloing paper:

Anthony Lomax, Alberto Michelini and Dario Jozinović, 2018. **An investigation of rapid earthquake characterization using single station waveforms and a convolutional neural network**, Seismological Research Letters


## Overview

ConvNetQuake_INGV, derived from ConvNetQuake (Perol et al., 2018; http://advances.sciencemag.org/content/4/2/e1700578.full), implements a CNN to characterize earthquakes at any distance (local to far teleseismic). ConvNetQuake_INGV operates on 50sec, 3-component, broadband, single-station waveforms to detect seismic events and obtain binned, probabilistic estimates of the distance, azimuth, depth and magnitude of the event. ConvNetQuake_INGV is trained through supervised learning using waveform windows containing a diverse set of known event and noise waveforms.

For ConvNetQuake_INGV, we modify the procedures and codes of Perol et al. (2018) and develop new tools to retrieve events, and noise and event waveforms from FDSN web-services (http://www.fdsn.org/webservices).

For details see The ConvNetQuake_INGV section below.


## Installation

    # install environment
    # https://www.tensorflow.org/install/install_mac
    sudo easy_install pip
    pip install --upgrade virtualenv 
    virtualenv --system-site-packages .
    source ./bin/activate
    easy_install -U pip
    pip install grpcio==1.9.1
    pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.6.0-py2-none-any.whl
    pip install -r requirements.txt  # needed?
    pip install -r requirements2.txt  # needed?
    pip install python-gflags
    pip install -e git+https://github.com/gem/oq-hazardlib.git@589fa31ddca3697e6b167082136dc0174a77bc16#egg=openquake.hazardlib
    pip install dask --upgrade
    pip install argparse
    pip install geojson
    pip install geopy
    pip install folium

    # install ConvNetQuake -> https://github.com/tperol/ConvNetQuake
    # link ConvNetQuake code in current working directory
    ln -s /Users/anthony/opt/ConvNetQuake-master_tf1.0/tflib .
    #
    # convert ConvNetQuake code to TensorFlow 1.2
    # upgrade a whole directory of 0.n TensorFlow programs to 1.0, enter a command having the following format:
    # $ python tf_upgrade.py --intree InputDir --outtree OutputDir
    # e.g.
    python tf_upgrade.py --intree ./tflib --outtree ./tflib
    chmod a+x ./tflib/*/*.py


    # run ConvNetQuake_INGV codes
    source ./bin/activate; export PYTHONPATH=./:./quakenet_ingv
    ...


## Run examples

    #install
    see INSTALLATION.txt

    # activate environment
    source ./bin/activate; export PYTHONPATH=./:./quakenet_ingv

    # get channel_file
    http://webservices.ingv.it/fdsnws/station/1/query?network=MN&channel=BH?&starttime=2009-01-01&endtime=2018-12-31&level=channel 
    # save as: data/MN/MN_stations_2009-2018.xml

    # get train and validation events for circular distances centred on each station in channel_file
    python quakenet_ingv/bin/preprocess/get_events.py --base_url IRIS --channel_file data/MN/MN_stations_2009-2018.xml --starttime 2010-01-01 --endtime 2018-12-31 --channel_prefix BH --sampling_rate 20 --minradius 0.0 --maxradius 2.0 --minmagnitude 3.0 --event_files_path output/MN3/events
    python quakenet_ingv/bin/preprocess/get_events.py --base_url IRIS --channel_file data/MN/MN_stations_2009-2018.xml --starttime 2010-01-01 --endtime 2018-12-31 --channel_prefix BH --sampling_rate 20 --minradius 2.0 --maxradius 20.0 --minmagnitude 4.0 --event_files_path output/MN3/events
    python quakenet_ingv/bin/preprocess/get_events.py --base_url IRIS --channel_file data/MN/MN_stations_2009-2018.xml --starttime 2010-01-01 --endtime 2018-12-31 --channel_prefix BH --sampling_rate 20 --minradius 20.0 --maxradius 70.0 --minmagnitude 5.0 --event_files_path output/MN3/events
    python quakenet_ingv/bin/preprocess/get_events.py --base_url IRIS --channel_file data/MN/MN_stations_2009-2018.xml --starttime 2010-01-01 --endtime 2018-12-31 --channel_prefix BH --sampling_rate 20 --minradius 70.0 --maxradius 180.0 --minmagnitude 6.0 --event_files_path output/MN3/events

    python quakenet_ingv/bin/preprocess/get_streams.py --event_files_path output/MN3/events --outpath output/MN3/streams --base_url INGV --channel_file data/MN/MN_stations_2009-2018.xml --channel_prefix BH --sampling_rate 20 --n_streams 5000 --noise_fraction 0.4 --validation_fraction 0.1 --test_fraction 0.0 --snr_accept 3.0 --window_length 50 --window_start 5 --n_distances 50 --n_magnitudes 20 --n_depths 20 --n_azimuths 36 --event_fraction 0.25

    # train
    quakenet_ingv/bin/train --outpath output/MN3/streams --dataset output/MN3/streams/train --model ConvNetQuake9 --checkpoint_dir output/MN3/ConvNetQuake9 --use_magnitudes --use_depths --use_azimuths
    tensorboard --logdir output/MN3/ConvNetQuake9/ConvNetQuake9
    quakenet_ingv/bin/evaluate --outpath output/MN3/streams --model ConvNetQuake9 --checkpoint_dir output/MN3/ConvNetQuake9/ConvNetQuake9 --dataset output/MN3/streams/validate --eval_interval -1 --use_magnitudes --use_depths --use_azimuths --events


    # get test events for circular distances centred on each station in channel_file
    python quakenet_ingv/bin/preprocess/get_events.py --base_url IRIS --channel_file data/MN/MN_stations_2009-2018.xml --starttime 2007-01-01 --endtime 2009-12-31 --channel_prefix BH --sampling_rate 20 --minradius 0.0 --maxradius 2.0 --minmagnitude 4.0 --event_files_path output/MN/test_events
    python quakenet_ingv/bin/preprocess/get_events.py --base_url IRIS --channel_file data/MN/MN_stations_2009-2018.xml --starttime 2007-01-01 --endtime 2009-12-31 --channel_prefix BH --sampling_rate 20 --minradius 2.0 --maxradius 20.0 --minmagnitude 5.0 --event_files_path output/MN/test_events
    python quakenet_ingv/bin/preprocess/get_events.py --base_url IRIS --channel_file data/MN/MN_stations_2009-2018.xml --starttime 2007-01-01 --endtime 2009-12-31 --channel_prefix BH --sampling_rate 20 --minradius 20.0 --maxradius 50.0 --minmagnitude 6.0 --event_files_path output/MN/test_events
    python quakenet_ingv/bin/preprocess/get_events.py --base_url IRIS --channel_file data/MN/MN_stations_2009-2018.xml --starttime 2007-01-01 --endtime 2009-12-31 --channel_prefix BH --sampling_rate 20 --minradius 50.0 --maxradius 180.0 --minmagnitude 6.5 --event_files_path output/MN/test_events
    #
    python quakenet_ingv/bin/preprocess/get_streams.py --event_files_path output/MN/test_events --outpath output/MN3/test_streams --base_url INGV --channel_file data/MN/MN_stations_2009-2018.xml --channel_prefix BH --sampling_rate 20 --n_streams 999 --noise_fraction 0.0 --validation_fraction 0.0 --test_fraction 1.0 --snr_accept -1.0 --window_length 50 --window_start 5 --n_distances 50 --n_magnitudes 20 --n_depths 20 --n_azimuths 36 --systematic
    #
    quakenet_ingv/bin/evaluate --outpath output/MN3/test_streams --model ConvNetQuake9 --checkpoint_dir output/MN3/ConvNetQuake9/ConvNetQuake9_NNN --dataset output/MN3/test_streams_YEARS/test --eval_interval -1 --use_magnitudes --use_depths --use_azimuths --channel_file data/MN/MN_stations_2007-2018.xml --events --save_event_results --step NNN
    # NOTE: move event_results to event_results_YEARS
    #
    ./run_event_results_batch.bash output/MN3/ConvNetQuake9/ConvNetQuake9_NNN/event_results_YEARS

    quakenet_ingv/bin/plot_events --event_files_path output/MN3/streams/train/events/xml --channel_file data/MN/MN_stations_2007-2018.xml --channel_prefix BH --sampling_rate 20 --n_magnitudes 20 --n_depths 20
    quakenet_ingv/bin/plot_events --event_files_path output/MN3/test_streams_2009/test/events/xml --channel_file data/MN/MN_stations_2007-2018.xml --channel_prefix BH --sampling_rate 20 --n_magnitudes 20 --n_depths 20

    quakenet_ingv/bin/plot_event_stats --evaluate_event_results_path output/MN3/ConvNetQuake9/ConvNetQuake9_NNN/event_results_YEARS --train_events_path output/MN3/streams/train/events/xml


## The ConvNetQuake_INGV algorithm

The procedures and Python codes (https://github.com/tperol/ConvNetQuake) of Perol et al. (2018; http://advances.sciencemag.org/content/4/2/e1700578.full) for ConvNetQuake are based on several technologies:

1. The Python (https://www.python.org) programming language, with the NumPy (http://www.numpy.org) package for scientific computing, and many other packages.
2. The ObsPy (http://obspy.org) Python framework for processing seismological data.
3. TensorFlow (Abadi et al. 2015; https://www.tensorflow.org), an open source software library for high performance numerical computation with strong support for machine learning and deep learning.

For ConvNetQuake_INGV, we modify the procedures and codes of Perol et al. (2018) and develop new Python tools to:

1. Retrieve events in QuakeML format from an FDSN “event” web-service for specified magnitude and distance ranges for each station in a QuakeML channel file, possibly retrieved from an FDSN “station” web-service. (quakenet_ingv/bin/preprocess/get_events.py)
2. Retrieve from an FDSN “dataselect” web-service noise and event waveforms (Figures 1 in the main text) for training, validation and test data sets with specified proportions of noise/event samples.  For the training and validation data sets, waveforms are obtained for randomly chosen stations from the QuakeML channel file and randomly chosen events from the QuakeML event files generated by get_events.py.  For the test data set, waveforms are obtained for all available stations in the QuakeML channel file and randomly chosen events from the QuakeML event files generated by get_events.py.  Waveforms are labelled with binned (Table S1) station-event distance, station-event azimuth, event magnitude and event depth. (quakenet_ingv/bin/preprocess/get_streams.py)
3. Set up the neural network architecture (Figure 2 and description in main text), define the loss function, setup the optimizer (inverts for network node weights and biases), and evaluate validation metrics. (quakenet_ingv/quakenet/models.py)
4. Set-up and train the neural network by feeding batches of training event and noise waveforms stored by get_streams.py to routines defined in models.py (quakenet_ingv/bin/train)
5. Evaluate results of network training by feeding specified validation event or noise waveforms stored by get_streams.py to trained network, outputs various metrics showing quality of match between predicted and true classification values for detection, station-event distance, azimuth, event magnitude and depth. (quakenet_ingv/bin/evaluate)
6. Plot Evaluate results output for the test data set using Folium (http://python-visualization.github.io/folium) to generate leaflet.js (https://leafletjs.com) html pages showing, for each event: 1) an interactive map of stations, true epicenter and predicted distance-azimuth bins with finite probability, 2) a histogram of true and predicted magnitude bins,  2) a histogram of true and predicted depth bins. (quakenet_ingv/bin/plot_event_results)

### Target waveform labels

The class labels or classification for each waveform are: event or noise, binned station-event distance (0-180°), station-event azimuth  (0-360°, 10° step), event magnitude (0-10, 0.5mu step) and event depth (0-700km).  The bin steps for distance and depth are geometrically increasing to give higher weight and precision to nearer and shallower events.  Table S1 lists the binned, target labels.  Following Perol et al. (2018), the event or noise label is a binary classification, it is implemented as a -1 value for the station-event distance label.

### Waveform datasets

The event and noise waveforms are identical except for being labelled as event or noise, and their data files are organized in different event or noise sub-directories in train, validate and test stream directories.

3-component (BHZ/N/E) waveforms are retrieved from an FDSN “dataselect” web-service files for specified proportions of noise/event and train/validate/test samples.    The training earthquake waveforms should span, as well as possible, the distance, azimuth, depth and magnitude range of target events for application of the trained ConvNetQuake_INGV. Waveforms are obtained for randomly chosen network-station specified in a QuakeML format channel file.  Then, to define a channel event waveforms, events are randomly chosen from QuakeML format event files generated by get_events.py.  For noise waveforms, a reference time is chosen between consecutive events randomly chosen from the QuakeML format event files.

Waveform quality control and pre-processing includes: 

1. checking that all 3 components are available, 
2. resampling to 20Hz if necessary,
3. checking that event waveforms have signal-to-noise ratio (SNR) greater than a specified threshold [3.0] for at least one broadband or 1Hz, 4-pole high-pass filtered component
4. checking that noise waveforms have signal-to-noise ratio (SNR) less than a specified threshold for all components in broadband or with 1Hz, 4-pole high-pass,
5. trim event waveforms to start at a specified time interval [5 sec] before predicted P arrival time (using ak135 model) and event and noise waveforms to a specified total window length [50 sec],
6. correct amplitudes to sensitivity (gain) for each component,
7. normalized to the global maximum of all 3 traces, store the normalization value,  stream_max.
8. for event waveforms, determine true binned distance, azimuth, magnitude and depth class.

Since the pre-processing includes normalized to the global maximum of all 3 traces, the normalization value, stream_max, is used by the neural network to provide absolute trace amplitude information to aid in magnitude estimation.

After quality control and pre-processing, the windowed noise and event waveforms are saved to the stream directories in miniseed format (IRIS, 2012) and, along with the true waveform classification value, in TensorFlow “TFRecord” format (https://www.tensorflow.org).  For event waveforms, event meta-data in QuakeML are also saved to the stream directories.

### Loss function and optimizer

ConvNetQuake_INGV uses an L2-regularized cross-entropy loss (misfit) function and the Adam Optimizer algorithm, as in ConvNetQuake (Perol et al., 2018; Mehta et al., 2018; Kingma and Ba, 2017).  The Adam Optimizer performs first-order, gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data or parameters.

