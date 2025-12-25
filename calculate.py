import spiceypy
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from product_parser import DsnStationAllocationFileDecoder

# derived from https://github.com/behrouzz/astronomy/blob/9fd270b74e609d149a615f49e0ecc729b1d6b360/spice/bspice.py
# also see example code from https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/azlcpo.html

d2r = np.pi / 180
r2d = 180 / np.pi

def calc_ltb_azel(et: float, dish_loc_cartesian: tuple[float, float, float] | np.ndarray, abcorr='CN+S') -> dict:
    """
    Calculate azimuth and elevation to lunar trailblazer spacecraft, given ground station location.
    et: J2000 time
    dish_loc_cartesian: Rectangular coordinates of ground station, Earth frame, km
    """

    state, lt  = spiceypy.azlcpo(
        method='ELLIPSOID',
        target='-242', # LTB spacecraft
        et=et,
        abcorr=abcorr,
        azccw=False,
        elplsz=True,
        obspos=dish_loc_cartesian,
        obsctr='earth',
        obsref='ITRF93')

    return {
        'Time (UTC)': spiceypy.et2utc(et, 'ISOC', 0),
        'Range (km)': state[0],
        'Azimuth (deg)': state[1] * r2d,
        'Elevation (deg)': state[2] * r2d,
        'd Range/dt (km/s)': state[3],
        'd Azimuth/dt (deg/s)': state[4] * r2d,
        'd Elevation/dt (deg/s)': state[5] * r2d,
        'One Way Light Time (s)': lt,
    }


def lonlat_to_cartesian(lon, lat, alt):
    """
    lon: Longitude (deg)
    lat: Latitude (def)
    alt: Altitude (m)
    Returns (x, y, z) in Earth frame, km
    """
    lon = lon * d2r
    lat = lat * d2r
    alt = alt / 1000

    dim, values = spiceypy.bodvrd(bodynm="earth", item="RADII", maxn=3)

    re  =  values[0]
    rp  =  values[2]
    # flattening coefficient
    f   =  ( re - rp ) / re

    return spiceypy.pgrrec(body='earth', lon=lon, lat=lat, alt=alt, re=re, f=f)


def main():
    parent_dir = Path(__file__).parent
    data_dir = parent_dir / 'data'
    out_dir = parent_dir / 'calculated'

    parser = argparse.ArgumentParser(description='Calculate station visibility for Lunar Trailblazer spacecraft. Writes CSV files to `calculated` directory.')
    parser.add_argument('--saf', default=str(data_dir / 'LTB_25062_25117E.SAF'), help='Path to Station Allocation File (SAF), such as LTB_25062_25117E.SAF')
    parser.add_argument('--bsp', default=str(data_dir / 'ltb_trj_od006v1.bsp'), help='Path to LTB trajectory kernel file, such as ltb_trj_od005v1.bsp')
    parser.add_argument('--start', default='2025-03-08T00:00:00Z')
    parser.add_argument('--end', default='2025-04-08T00:00:00Z')
    args = parser.parse_args()

    spiceypy.furnsh(str(args.bsp))
    spiceypy.furnsh(str(data_dir / 'naif0012.tls'))
    spiceypy.furnsh(str(data_dir / 'pck00010.tpc'))
    spiceypy.furnsh(str(data_dir / 'earth_latest_high_prec.bpc'))
    spiceypy.furnsh(str(data_dir / 'earthstns_itrf93_201023.bsp'))

    dishes = pd.read_csv(data_dir / 'dishes.csv')

    start = spiceypy.utc2et(args.start)
    end = spiceypy.utc2et(args.end)
    spacing = 60

    out_dir.mkdir(exist_ok=True)

    for row in dishes.iloc:
        print('calculate', row['Location'])
        dish_loc_cartesian = lonlat_to_cartesian(row['Longitude (deg)'], row['Latitude (deg)'], row['Altitude (m)'])

        out_rows = []
        et = start
        while et < end:
            out_rows.append(calc_ltb_azel(et, dish_loc_cartesian))
            et += spacing
        
        out_name = row['Location'].replace(' ', '_').replace(',', '')
        out_path = out_dir / f'{out_name}.csv'
        pd.DataFrame(out_rows).to_csv(out_path, index=False)
    
    saf_parser = DsnStationAllocationFileDecoder(filename=args.saf)

    # merge tracks with same antenna into one CSV
    dss_antenna_rows = {}

    for allocation in saf_parser.parse():
        antenna_id, bot, eot = allocation['ANTENNA_ID'], allocation['BOT'], allocation['EOT']

        print(f'calculate {antenna_id}, BOT {bot} -> EOT {eot}')

        bot_et = spiceypy.datetime2et(bot)
        eot_et = spiceypy.datetime2et(eot)
        
        out_rows = dss_antenna_rows.get(antenna_id, [])
        et = bot_et
        while et < eot_et:
            dish_loc_cartesian = spiceypy.spkpos(targ=antenna_id, et=et, ref='ITRF93', abcorr='NONE', obs='EARTH')[0]
            out_rows.append(calc_ltb_azel(et, dish_loc_cartesian))
            et += spacing
        # one more entry at end of track, if different from last time point
        if eot_et - et > 1:
            dish_loc_cartesian = spiceypy.spkpos(targ=antenna_id, et=eot_et, ref='ITRF93', abcorr='NONE', obs='EARTH')[0]
            out_rows.append(calc_ltb_azel(eot_et, dish_loc_cartesian))

        dss_antenna_rows[antenna_id] = out_rows
    
    for antenna_id, out_rows in dss_antenna_rows.items():
        pd.DataFrame(out_rows).to_csv(out_dir / f'{antenna_id}.csv', index=False)

if __name__ == '__main__':
    main()
