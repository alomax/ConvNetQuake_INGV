'''
Created on 29 Mar 2018

@author: Anthony Lomax - ALomax Scientific
'''

"""Get streams from FDSN web service."""

import os
import traceback
import argparse
import random
from datetime import datetime
import cPickle as pickle

import math
import numpy as np

from obspy import read_inventory
from obspy.core.event import Catalog
from obspy.core.event.catalog import read_events
import obspy.clients.fdsn as fdsn
from  obspy.clients.fdsn.header import (DEFAULT_PARAMETERS)
from obspy.taup import TauPyModel
import obspy.geodetics.base as geo

import quakenet.util as util
from quakenet.data_pipeline import DataWriter


WINDOW_PADDING_FDSN = 60.0  # pad requested time window to avoid truncated waveforms
MIN_INTER_EVENT_TIME = 60.0*120.0  # 2 hours, minimum inter event time in seconds to process for noise waveform
MAX_SNR_NOISE = 1.5
TAU_PY_MODEL = TauPyModel(model='ak135')



def get_first_P_travel_time(origin, channel):
    
    # get first P arrival travel-time
    arrivals = TAU_PY_MODEL.get_travel_times_geo( \
                                   origin.depth / 1000.0, origin.latitude, origin.longitude, \
                                   channel.latitude, channel.longitude, \
                                   phase_list=('ttp', ))
    
    print 'arrivals', arrivals
    
    ttime = arrivals[0].time
    if arrivals[0].name == 'Pdiff' and len(arrivals) > 1:
        ttime = arrivals[1].time
        print 'using ', arrivals[1].name
    else:
        print 'using ', arrivals[0].name
    
    return ttime


def   get_systematic_channel(inventory, catalog_all, is_noise, event_ndx, net_ndx, sta_ndx, channel_ndx):

    print 'get_systematic_channel', 'event_ndx', event_ndx, 'net_ndx', net_ndx, 'sta_ndx', sta_ndx, 'channel_ndx', channel_ndx

    net = inventory[net_ndx]
    sta = net[sta_ndx]
    channel_ndx = channel_ndx + 1
    if channel_ndx >= len(sta):
        channel_ndx = 0
        sta_ndx = sta_ndx + 1
        if sta_ndx >= len(net):
            sta_ndx = 0
            net_ndx = net_ndx + 1
            if net_ndx >= len(inventory):
                net_ndx = 0
                event_ndx = event_ndx + 1
                if event_ndx >= len(catalog_all):
                    raise ValueError("finished processing inventory and events")
            net = inventory[net_ndx]
        sta = net[sta_ndx]
    channel = sta[channel_ndx]
        
    event = catalog_all[event_ndx]
    origin = event.preferred_origin()
                
    return catalog_all, event_ndx, event, origin, channel, net_ndx, net, sta_ndx, sta, channel_ndx


def   get_random_channel(inventory, catalog_dict, is_noise):

    nnet = len(inventory)
    if nnet > 0:
        net_ndx = random.randint(0, nnet - 1)
        net = inventory[net_ndx]
        nsta = len(net)
        if nsta > 0:
            sta_ndx = random.randint(0, len(net) - 1)
            sta = net[sta_ndx]
            net_sta = net.code + '_' + sta.code
            if not net_sta in catalog_dict:
                raise ValueError("net_sta not in catalog_dict")
            nchans = len(sta)
            if nchans > 0:
                channel_ndx = random.randint(0, nchans - 1)
                channel = sta[channel_ndx]

                # get random event from Catalog
                catalog = catalog_dict[net_sta]
                imax = catalog.count() - 1
                if is_noise:
                    imax = catalog.count() - 2
                event_ndx = random.randint(0, imax)
                event = catalog[event_ndx]
                origin = event.preferred_origin()
                
    return catalog, event_ndx, event, origin, channel, net_ndx, net, sta_ndx, sta, channel_ndx



def main(args):
    
    random.seed(datetime.now())
    
    if args.n_distances < 1:
        args.n_distances = None
    # print distance classifications
    if args.n_distances != None:
        print 'dist_class, dist_deg, dist_km'
        for dclass in range(0, args.n_distances, 1):
            dist_deg = util.classification2distance(dclass, args.n_distances)
            dist_km = geo.degrees2kilometers(dist_deg)
            print "{}   {:.2f}   {:.1f}".format(dclass, dist_deg, dist_km)
        print ''
     
    if args.n_magnitudes < 1:
        args.n_magnitudes = None
    # print magtitude classifications
    if args.n_magnitudes != None:
        print 'mag_class, mag'
        for mclass in range(0, args.n_magnitudes, 1):
            mag = util.classification2magnitude(mclass, args.n_magnitudes)
            print "{}   {:.2f}".format(mclass, mag)
        print ''
     
    if args.n_depths < 1:
        args.n_depths = None
    # print depth classifications
    if args.n_depths != None:
        print 'depth_class, depth'
        for dclass in range(0, args.n_depths, 1):
            depth = util.classification2depth(dclass, args.n_depths)
            print "{}   {:.1f}".format(dclass, depth)
        print ''
     
    if args.n_azimuths < 1:
        args.n_azimuths = None
    # print azimuth classifications
    if args.n_azimuths != None:
        print 'azimuth_class, azimuth'
        for aclass in range(0, args.n_azimuths, 1):
            azimuth = util.classification2azimuth(aclass, args.n_azimuths)
            print "{}   {:.1f}".format(aclass, azimuth)
        print ''
     
    
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
        
    # save arguments
    with open(os.path.join(args.outpath, 'params.pkl'), 'w') as file:
        file.write(pickle.dumps(args)) # use `pickle.loads` to do the reverse
        
    for dataset in ['train', 'validate', 'test']:
        for datatype in ['events', 'noise']:
            datapath = os.path.join(args.outpath, dataset, datatype)
            if not os.path.exists(datapath):
                os.makedirs(datapath)
            mseedpath = os.path.join(datapath, 'mseed')
            if not os.path.exists(mseedpath):
                os.makedirs(mseedpath)
            mseedpath = os.path.join(datapath, 'mseed_raw')
            if not os.path.exists(mseedpath):
                os.makedirs(mseedpath)
            if datatype == 'events':
                xmlpath = os.path.join(datapath, 'xml')
                if not os.path.exists(xmlpath):
                    os.makedirs(xmlpath)

        
    # read catalog of events
    #filenames = args.event_files_path + os.sep + '*.xml'
    catalog_dict = {}
    catalog_all = []
    for dirpath, dirnames, filenames in os.walk(args.event_files_path):
        for name in filenames:
            if name.endswith(".xml"):
                file = os.path.join(dirpath, name)
                catalog = read_events(file)
                target_count = int(args.event_fraction * float(catalog.count()))
                print catalog.count(), 'events:', 'read from:', file, 'will use:', target_count, 'since args.event_fraction=', args.event_fraction
                if (args.event_fraction < 1.0):
                    while catalog.count() > target_count:
                        del catalog[random.randint(0, catalog.count() - 1)]
                if not args.systematic:
                    tokens = name.split('_')
                    net_sta = tokens[0] + '_' + tokens[1]
                    if not net_sta in catalog_dict:
                        catalog_dict[net_sta] = catalog
                    else:
                        catalog_dict[net_sta] += catalog
                    # sort catalog by date
                    catalog_dict[net_sta] = Catalog(sorted(catalog_dict[net_sta], key=lambda e: e.origins[0].time))
                else:
                    catalog_all += catalog
    
    # read list of channels to use
    inventory_full = read_inventory(args.channel_file)
    inventory_full = inventory_full.select(channel=args.channel_prefix+'Z', sampling_rate=args.sampling_rate)
    #print(inventory)
    
    client = fdsn.Client(args.base_url)
    
    # get existing already processed event channel dictionary
    try:
        with open(os.path.join(args.outpath, 'event_channel_dict.pkl'), 'r') as file:
            event_channel_dict = pickle.load(file)
    except IOError:
        event_channel_dict = {}
    print 'Existing event_channel_dict size:', len(event_channel_dict)

    n_noise = int(0.5 + float(args.n_streams) * args.noise_fraction)
    n_events = args.n_streams - n_noise
    n_validate = int(0.5 + float(n_events) * args.validation_fraction)
    n_test = int(0.5 + float(n_events) * args.test_fraction)
    n_train = n_events - n_validate - n_test
    n_count = 0;
    n_streams = 0
    
    if args.systematic:
        event_ndx = 0
        net_ndx = 0
        sta_ndx = 0
        channel_ndx = -1


    
#     distance_id_count = {}
#     max_num_for_distance_id = {}
#     if args.n_distances != None:
#         # train
#         distance_id_count['train'] = [0] * args.n_distances
#         max_num_for_distance_id['train'] = 1 + int(2.0 * float(n_train) / float(args.n_distances))
#         print 'Maximum number events for each distance bin train:', max_num_for_distance_id['train']
#         # validate
#         distance_id_count['validate'] = [0] * args.n_distances
#         max_num_for_distance_id['validate'] = 1 + int(2.0 * float(n_validate) / float(args.n_distances))
#         print 'Maximum number events for each distance bin validate:', max_num_for_distance_id['validate']
#         # test
#         distance_id_count['test'] = [0] * args.n_distances
#         max_num_for_distance_id['test'] = 1 + int(2.0 * float(n_test) / float(args.n_distances))
#         print 'Maximum number events for each distance bin test:', max_num_for_distance_id['test']
        
    while args.systematic or n_streams < args.n_streams:
        
        try:
        
            # choose event or noise
            is_noise = n_streams >= n_events
            
            # reset validate test count if switching from event to  noise
            if n_streams == n_events:
                n_validate = int(0.5 + float(n_noise) * args.validation_fraction)
                n_test = int(0.5 + float(n_noise) * args.test_fraction)
                n_train = n_noise - n_validate - n_test
                n_count = 0;
                
            # set out paths
            if is_noise:
                datatype = 'noise'
            else:
                datatype = 'events'
            if n_count < n_train:
                dataset = 'train'
            elif n_count < n_train + n_validate:
                dataset = 'validate'
            else:
                dataset = 'test'
            datapath = os.path.join(args.outpath, dataset, datatype)

            # get random channel from Inventory
            #inventory = inventory_full.select(time=origin.time)
            inventory = inventory_full
            
            if args.systematic:
                try:
                    catalog, event_ndx, event, origin, channel, net_ndx, net, sta_ndx, sta, channel_ndx \
                        = get_systematic_channel(inventory, catalog_all, is_noise, event_ndx, net_ndx, sta_ndx, channel_ndx)
                except ValueError:
                    break
            else:
                try:
                    catalog, event_ndx, event, origin, channel, net_ndx, net, sta_ndx, sta, channel_ndx = get_random_channel(inventory, catalog_dict, is_noise)
                except ValueError:
                    continue
                                
            distance_id = 0
            distance = -999.0
            magnitude = -999.0
            depth = -999.0
            azimuth = -999.0
            if not is_noise:
                dist_meters, azim, bazim = geo.gps2dist_azimuth(channel.latitude, channel.longitude, origin.latitude, origin.longitude, a=geo.WGS84_A, f=geo.WGS84_F)
                distance = geo.kilometer2degrees(dist_meters / 1000.0, radius=6371)
                azimuth = azim
                magnitude = event.preferred_magnitude().mag
                depth = origin.depth / 1000.0
                if args.n_distances != None:
                    distance_id = util.distance2classification(distance, args.n_distances)
#                                 if distance_id_count[dataset][distance_id] >= max_num_for_distance_id[dataset]:
#                                     print 'Skipping event_channel: distance bin', distance_id, 'for', dataset, 'already full:', \
#                                         distance_id_count[dataset][distance_id], '/', max_num_for_distance_id[dataset]
#                                     continue

            print ''
            print 'Event:', origin.time.isoformat(), event.event_descriptions[0].text, \
            ', Dist(deg): {:.2f} Dist(km): {:.1f} ID: {}'.format(distance, geo.degrees2kilometers(distance), distance_id), \
            ', Mag: {:.2f}'.format(magnitude), \
            ', Depth(km): {:.1f}'.format(depth), \
            ', Az(deg): {:.1f}'.format(azimuth)
            print 'Retrieving channels:', (n_streams + 1), '/ ', args.n_streams, (', NOISE, ' if  is_noise else ', EVENT, '), 'event', event_ndx, origin.time, \
                ', net', net_ndx, ', sta', sta_ndx, ', chan', channel_ndx, \
                ', ', net.code, sta.code, \
                channel.code, channel.location_code, \
                channel.sample_rate
            # check station was available at origin.time
            if not sta.is_active(time=origin.time):
                print 'Skipping event_channel: station not active at origin.time:'
                continue
            #key = str(event_ndx) + '_' + str(net_ndx) + '_' + str(sta_ndx) + '_' + str(channel_ndx) + '_' + str(is_noise)
            key = str(event_ndx) + '_' + net.code + '_' + sta.code + '_' + channel.code + '_' + str(is_noise)
            if key in event_channel_dict:
                print 'Skipping event_channel: already processed.'
                continue
            event_channel_dict[key] = 1
                
            # get start time for waveform request
            ttime = get_first_P_travel_time(origin, channel)
            arrival_time = origin.time + ttime
            if is_noise:
                # get start time of next event
                event2 = catalog[event_ndx + 1]
                origin2 = event2.preferred_origin()
                # check that origins are at least min time apart
                if origin2.time - origin.time < MIN_INTER_EVENT_TIME:
                    print 'Skipping noise event_channel: inter event time too small: ', str(origin2.time - origin.time), \
                        origin2.time, origin.time
                    continue
                ttime2 = get_first_P_travel_time(origin2, channel)
                arrival_time2 = origin2.time + ttime2
                arrival_time = (arrival_time + ((arrival_time2 - arrival_time) / 2.0)) - args.window_start
            
            start_time = arrival_time - args.window_start
                                    
            # request data for 3 channels
            
            #for orientation in ['Z', 'N', 'E', '1', '2']:
            #    req_chan = args.channel_prefix + orientation
            channel_name = net.code + '_' + sta.code + '_' + channel.location_code + '_' + args.channel_prefix
            padded_start_time = start_time - WINDOW_PADDING_FDSN
            padded_end_time = start_time + args.window_length + 2.0 * WINDOW_PADDING_FDSN
            chan_param = args.channel_prefix + '?'
            # kluge to get url used for data request
            kwargs = {'network': net.code, 'station': sta.code, 'location': channel.location_code, 'channel': chan_param,
                      'starttime': padded_start_time, 'endtime': padded_end_time}                      
            #url = client._create_url_from_parameters('dataselect', DEFAULT_PARAMETERS['dataselect'],  **kwargs)
            url = fdsn.client.build_url(client.base_url, 'dataselect', client.major_versions['dataselect'], "query", parameters=kwargs)
            print '  java net.alomax.seisgram2k.SeisGram2K', '\"', url, '\"'
            try:
                stream = client.get_waveforms(  \
                                               net.code, sta.code, channel.location_code, chan_param, \
                                               padded_start_time, padded_end_time, \
                                               attach_response=True)
                
            except fdsn.header.FDSNException as ex:
                print 'Skipping channel:', channel_name, 'FDSNException:', ex, 
                continue
                                    
            print stream
            # TEST
#                         for trace in stream:
#                             print '==========> trace.stats', trace.stats
                
            # check some things
            if (len(stream) != 3):
                print 'Skipping channel: len(stream) != 3:', channel_name
                continue
            ntrace = 0
            for trace in stream:
                if (len(trace) < 1):
                    print 'Skipping trace: len(trace) < 1:', channel_name
                    continue
                if (trace.stats.starttime > start_time or trace.stats.endtime < start_time + args.window_length):
                    print 'Skipping trace: does not contain required time window:', channel_name
                    continue
                ntrace += 1
            if (ntrace != 3):
                print 'Skipping channel: ntrace != 3:', channel_name
                continue
            
            # pre-process streams
            # sort so that channels will be ingested in NN always in same order ENZ
            stream.sort(['channel'])
            # detrend - this is meant to be equivalent to detrend or a long period low-pass (e.g. at 100sec) applied to real-time data
            stream.detrend(type='linear')
            for trace in stream:
                # correct for required sampling rate
                if abs(trace.stats.sampling_rate - args.sampling_rate) / args.sampling_rate > 0.01:
                    trace.resample(args.sampling_rate)
                    
            # apply high-pass filter if requested
            if args.hp_filter_freq > 0.0:
                stream.filter('highpass', freq=args.hp_filter_freq, corners=args.hp_filter_corners)
            
            # check signal to noise ratio, if fail, repeat on 1sec hp data to capture local/regional events in longer period microseismic noise
            sn_type = 'BRB'
            first_pass = True;
            while True:
                if is_noise:
                    snrOK = True
                else:
                    snrOK = False
                for trace in stream:
                    # slice with 1sec margin of error for arrival time to: 1) avoid increasing noise amplitude with signal, 2) avoid missing first P in signal
                    if (first_pass):
                        signal_slice = trace.slice(starttime=arrival_time - 1.0, endtime=arrival_time - 1.0 + args.snr_window_length)
                        noise_slice = trace.slice(endtime=arrival_time - 1.0) 
                    else:
                        # highpass at 1sec
                        filt_trace = trace.copy()
                        filt_trace.filter('highpass', freq=1.0, corners=4)
                        signal_slice = filt_trace.slice(starttime=arrival_time - 1.0, endtime=arrival_time - 1.0 + args.snr_window_length)
                        noise_slice = filt_trace.slice(endtime=arrival_time - 1.0) 
                        sn_type = '1HzHP'
                    # check signal to noise around arrival_time
                    # ratio of std
                    asignal = signal_slice.std()
                    anoise = noise_slice.std()
                    snr = asignal / anoise
                    print trace.id, sn_type, 'snr:', snr, 'std_signal:', asignal, 'std_noise:', anoise
                    # ratio of peak amplitudes (DO NOT USE, GIVE UNSTABLE RESULTS!)
#                                 asignal = signal_slice.max()
#                                 anoise = noise_slice.max()
#                                 snr = np.absolute(asignal / anoise)
#                                 print trace.id, sn_type, 'snr:', snr, 'amax_signal:', asignal, 'amax_noise:', anoise
                    if is_noise:
                        snrOK = snrOK and snr <= MAX_SNR_NOISE
                        if not snrOK:
                            break
                    else:
                        snrOK = snrOK or snr >= args.snr_accept
                if (first_pass and not snrOK and args.hp_filter_freq < 0.0):
                    first_pass = False;
                    continue
                else:
                    break

            if (not snrOK):
                if is_noise:
                    print 'Skipping channel:', sn_type, 'snr >', MAX_SNR_NOISE,  'on one or more traces:', channel_name
                else:
                    print 'Skipping channel:', sn_type, 'snr < args.snr_accept:', args.snr_accept, 'on all traces:', channel_name
                continue
               
            # trim data to required window
            # try to make sure samples and start/end times align as closely as possible to first trace
            trace = stream.traces[0]
            trace = trace.slice(starttime=start_time, endtime=start_time + args.window_length, nearest_sample=True)
            start_time = trace.stats.starttime
            stream = stream.slice(starttime=start_time, endtime=start_time + args.window_length, nearest_sample=True)
            
            cstart_time = '%04d.%02d.%02d.%02d.%02d.%02d.%03d' % \
                (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, \
                 start_time.second, start_time.microsecond // 1000)

            # process each trace
            try:
                for trace in stream:
                    # correct for overall sensitivity or gain
                    trace.normalize(trace.stats.response.instrument_sensitivity.value)
                    trace.data = trace.data.astype(np.float32)
                    # write miniseed
                    #tracefile = os.path.join(datapath, 'mseed', trace.id + '.' + cstart_time + '.mseed')
                    #trace.write(tracefile, format='MSEED', encoding='FLOAT32')
                    #print 'Channel written:', tracefile, trace.count(), 'samples'
            except AttributeError as err:
                print 'Skipping channel:', channel_name,  ': Error applying trace.normalize():' , err
                
            filename_root =  channel_name + '.' + cstart_time

            # write raw miniseed
            streamfile = os.path.join(datapath, 'mseed_raw', filename_root + '.mseed')
            stream.write(streamfile, format='MSEED', encoding='FLOAT32')
            print 'Stream written:', stream.count(), 'traces:'
            print '  java net.alomax.seisgram2k.SeisGram2K', streamfile
                
            # store absolute maximum
            stream_max = np.absolute(stream.max()).max()
            # normalize by absolute maximum
            stream.normalize(global_max = True)
            
            # 20180521 AJL
            # spherical coordinates
            # raw data always in same order ENZ
            # tensor indexing is [traces, datapoints, comps]
            if args.spherical:
                rad2deg = 180.0 / math.pi
                # calculate modulus
                temp_square = np.add(np.square(stream.traces[0].data), np.add(np.square(stream.traces[1].data), np.square(stream.traces[2].data)))
                temp_modulus = np.sqrt(temp_square)
                # calculate azimuth
                temp_azimuth = np.add( np.multiply(np.arctan2(stream.traces[0].data, stream.traces[1].data), rad2deg), 180.0)
                # calculate inclination
                temp_inclination = np.multiply(np.arcsin(np.divide(stream.traces[2].data, temp_modulus)), rad2deg)
                # reset stream data to spherical coordinates
                stream.traces[0].data = temp_inclination
                stream.traces[1].data = temp_azimuth
                temp_modulus = np.multiply(temp_modulus, 100.0)  # increase scale for plotting purposes
                stream.traces[2].data = temp_modulus


            # put absolute maximum normalization in first element of data array, to seed NN magnitude estimation
            # 20180816 AJL - do not mix max with data
            # for trace in stream:
            #    trace.data[0] = stream_max
            print 'stream_max', stream_max
            

            # write processed miniseed
            streamfile = os.path.join(datapath, 'mseed', filename_root + '.mseed')
            stream.write(streamfile, format='MSEED', encoding='FLOAT32')
            print 'Stream written:', stream.count(), 'traces:'
            print '  java net.alomax.seisgram2k.SeisGram2K', streamfile
                
            # write event waveforms and distance_id in .tfrecords
            magnitude_id = 0
            depth_id = 0
            azimuth_id = 0
            if not is_noise:
#                             if args.n_distances != None:
#                                 distance_id_count[dataset][distance_id] += 1
                if args.n_magnitudes != None:
                    magnitude_id = util.magntiude2classification(magnitude, args.n_magnitudes)
                if args.n_depths != None:
                    depth_id = util.depth2classification(depth, args.n_depths)
                if args.n_azimuths != None:
                    azimuth_id = util.azimuth2classification(azimuth, args.n_azimuths)
            else:
                distance_id = -1
                distance = 0.0
            output_name = filename_root + '.tfrecords'
            output_path = os.path.join(datapath, output_name)
            writer = DataWriter(output_path)
            writer.write(stream, stream_max, distance_id, magnitude_id, depth_id, azimuth_id, distance, magnitude, depth, azimuth)
            if not is_noise:
                print '==== Event stream tfrecords written:', output_name, \
                'Dist(deg): {:.2f} Dist(km): {:.1f} ID: {}'.format(distance, geo.degrees2kilometers(distance), distance_id), \
                ', Mag: {:.2f} ID: {}'.format(magnitude, magnitude_id), \
                ', Depth(km): {:.1f} ID: {}'.format(depth, depth_id), \
                ', Az(deg): {:.1f} ID: {}'.format(azimuth, azimuth_id)
            else:
                print '==== Noise stream tfrecords written:', output_name, 'ID: Dist {}, Mag {}, Depth {}, Az {}'.format(distance_id, magnitude_id, depth_id, azimuth_id)
                
            # write event data
            if not is_noise:
                filename = os.path.join(datapath, 'xml', filename_root + '.xml')
                event.write(filename, 'QUAKEML')
           
            n_streams += 1
            n_count += 1
                    
        except KeyboardInterrupt:
            print 'Stopping: KeyboardInterrupt'
            break

        except Exception as ex:
            print 'Skipping stream: Exception:', ex
            traceback.print_exc()
            continue

    print n_streams, 'streams:', 'written to:', args.outpath

    # save event_channel_dict
    with open(os.path.join(args.outpath, 'event_channel_dict.pkl'), 'w') as file:
        file.write(pickle.dumps(event_channel_dict))

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--event_files_path', type=str,
                        help='Path to the directory of event QuakeML event xml files to process')
    parser.add_argument('--event_fraction', type=float, default=1.0,
                        help='Fraction of events to use, randomly removes events to achieve this fraction of events in xml files')
    parser.add_argument('--base_url', type=str, default='IRIS',
                        help='Base URL of FDSN web service or key string for recognized server')
    parser.add_argument('--channel_file', type=str,
                        help='File containing FDSNStationXML list of net/station/location/channel to retrieve')
    parser.add_argument('--channel_prefix', type=str,
                        help='Prefix of the channel code to retrieve (e.g. BH)')
    parser.add_argument('--sampling_rate', type=float,
                        help='Channel sample rate to retrieve (e.g. 20)')
    parser.add_argument('--n_streams', type=int,
                        help='Number of streams to retrieve')
    parser.add_argument('--noise_fraction', type=float,
                        help='Fraction of streams that are noise')
    parser.add_argument('--validation_fraction', type=float,
                        help='Fraction of streams that are validation data')
    parser.add_argument('--test_fraction', type=float,
                        help='Fraction of streams that are testing data')
    parser.add_argument('--window_start', type=float, default=1.0,
                        help='Start time before first-arriving P/PKP of the trace window (in seconds)')
    parser.add_argument('--window_length', type=float, default=50, 
                        help='Length of the trace window (in seconds)')
    parser.add_argument('--snr_accept', type=float, default=3.0, 
                        help='Minimum acceptable SNR')
    parser.add_argument('--snr_window_length', type=float, default=10, 
                        help='Length of the window after arrival for SNR (in seconds)')
    parser.add_argument('--outpath', type=str,
                        help='Path for stream output')
    
    parser.add_argument('--hp_filter_freq', type=float, default=-1.0,
                        help='High-pass filter corner frequency (in Hz)')
    parser.add_argument('--hp_filter_corners', type=int, default=4,
                        help='High-pass filter number of corners (poles)')
    
    parser.add_argument('--spherical', action='store_true')
    parser.set_defaults(spherical=False)
    
    parser.add_argument('--n_distances', type=int, default=None,
                        help='Number of distance classes, 0=none')
    parser.add_argument('--n_magnitudes', type=int, default=None,
                        help='Number of magnitude classes, 0=none')
    parser.add_argument('--n_azimuths', type=int, default=None,
                        help='Number of azimuth classes, 0=none')
    parser.add_argument('--n_depths', type=int, default=None,
                        help='Number of depth classes, 0=none')
    parser.add_argument('--systematic', action='store_true', help='Attempt to get up to --n_streams streams for all net/sta/chan')
    parser.set_defaults(systematic=False)


    args = parser.parse_args()

    main(args)
