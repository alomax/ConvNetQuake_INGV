'''
Created on 28 Mar 2018

@author: Anthony Lomax - ALomax Scientific
'''
from obspy.core.utcdatetime import UTCDateTime

"""Get events from FDSN web service."""

import os
import argparse

from obspy import read_inventory
import obspy.clients.fdsn as fdsn


def main(args):
    
    dirname = args.event_files_path
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
    # read list of channels to use
    inventory = read_inventory(args.channel_file)
    inventory = inventory.select(channel=args.channel_prefix+'Z', sampling_rate=args.sampling_rate)
    
    depth_str = '' if  (args.mindepth == None and args.maxdepth == None) else \
        (str(args.mindepth) if args.mindepth != None else (str(-999.9)  + '-' + str(args.maxdepth)  if args.maxdepth != None else '')) + 'km_'
    filename_base = '' \
        +  (str(args.minradius) if args.minradius != None else str(0.0)) + '-' + str(args.maxradius) + 'deg_' \
        +  depth_str \
        +  'M' + str(args.minmagnitude) + '-' \
            + (str(args.maxmagnitude) if args.maxmagnitude != None else '')
        
    print 'filename_base', filename_base
    
    events_starttime = UTCDateTime(args.starttime)
    events_endtime = UTCDateTime(args.endtime)
    print 'events_starttime', events_starttime
    print 'events_endtime', events_endtime
        
    for net in inventory:
        for sta in net:            
            outfile = os.path.join(args.event_files_path, net.code + '_' + sta.code + '_'  \
                                   + filename_base + '.xml')
            client = fdsn.Client(args.base_url)
            print 'net_sta', net.code + '_' + sta.code
            print 'sta.start_date', sta.start_date
            print 'sta.end_date', sta.end_date
            tstart = sta.start_date if sta.start_date > events_starttime else events_starttime
            tend = sta.end_date if sta.end_date < events_endtime else events_endtime
            print 'tstart', tstart
            print 'tend', tend
            if not tstart < tend:
                continue
            try:
                catalog = client.get_events(latitude=sta.latitude, longitude=sta.longitude, \
                                   starttime=tstart, endtime=tend, \
                                   minradius=args.minradius, maxradius=args.maxradius, \
                                   mindepth=args.mindepth, maxdepth=args.maxdepth, \
                                   minmagnitude=args.minmagnitude, maxmagnitude=args.maxmagnitude, \
                                   includeallorigins=False, includeallmagnitudes= False, includearrivals=False)
            except Exception as ex:
                print 'Skipping net:', net.code, 'sta:', sta.code, 'Exception:', ex, 
                continue
            #, filename=args.outfile)
            catalog.write(outfile, 'QUAKEML')
            print catalog.count(), 'events:', 'written to:', outfile
            #catalog.plot(outfile=outfile + '.png')

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_url', type=str, default='IRIS',
                        help="Base URL of FDSN web service or key string for recognized server")
    parser.add_argument('--starttime', type=str, default=None,
                        help='Limit to events on or after the specified start time')
    parser.add_argument('--endtime', type=str, default=None,
                        help='Limit to events on or before the specified end time')
    parser.add_argument('--channel_file', type=str,
                        help='File containing FDSNStationXML list of net/station/location/channel to retrieve')
    parser.add_argument('--channel_prefix', type=str,
                        help='Prefix of the channel code to retrieve (e.g. BH)')
    parser.add_argument('--sampling_rate', type=float,
                        help='Channel sample rate to retrieve (e.g. 20)')
    parser.add_argument('--minradius', type=float, default=None,
                        help='Limit to events within the specified minimum number of degrees from the geographic point defined by the latitude and longitude parameters')
    parser.add_argument('--maxradius', type=float,
                        help='Limit to events within the specified maximum number of degrees from the geographic point defined by the latitude and longitude parameters')
    parser.add_argument('--mindepth', type=float, default=None,
                        help='Limit to events with depth, in kilometers, larger than the specified minimum')
    parser.add_argument('--maxdepth', type=float, default=None,
                        help='Limit to events with depth, in kilometers, larger than the specified maximum')
    parser.add_argument('--minmagnitude', type=float,
                        help='Limit to events with a magnitude larger than the specified minimum')
    parser.add_argument('--maxmagnitude', type=float, default=None,
                        help='Limit to events with a magnitude larger than the specified maximum')
    parser.add_argument('--event_files_path', type=str,
                        help='Path to the directory to write event QuakeML event xml files')

    args = parser.parse_args()

    main(args)
