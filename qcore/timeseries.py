"""
Shared functions to work on time-series.

@author Viktor Polak
@date 13/09/2016

Patched 2026: LFSeis uses new parser, HFSeis/BBSeis hardened for numpy 1.x+
"""

import base64
from glob import glob
from io import BytesIO
import math
import os
from pathlib import Path
from subprocess import Popen, PIPE
from typing import Union, Set

from qcore.constants import MAXIMUM_EMOD3D_TIMESHIFT_1_VERSION
from qcore.formats import load_e3d_par
from qcore.utils import compare_versions

# Import the new parser
from qcore.timeseries_2026 import read_lfseis_directory

try:
    from scipy.signal import butter, resample
except ImportError:
    print("SciPy not installed. Certain functions will fail.")
try:
    butter
    from scipy.signal import sosfiltfilt
except NameError:
    pass
except ImportError:
    from qcore.sosfiltfilt import sosfiltfilt
import numpy as np

rfft = np.fft.rfft
irfft = np.fft.irfft

# ============================================================================
# UTILITY FUNCTIONS (unchanged)
# ============================================================================

def bwfilter(data, dt, freq, band, match_powersb=True):
    """
    data: np.array to filter
    dt: timestep of data
    freq: cutoff frequency
    band: One of {'highpass', 'lowpass', 'bandpass', 'bandstop'}
    match_powersb: shift the target frequency so that the given frequency has no attenuation
    """
    nyq = 1.0 / (2.0 * dt)
    highpass_shift = 0.8956803352330285
    lowpass_shift = 1.1164697500474103
    if match_powersb:
        if band == "highpass":
            freq *= highpass_shift
        elif band == "bandpass" or band == "bandstop":
            freq = np.asarray((freq * highpass_shift, freq * lowpass_shift))
        elif band == "lowpass":
            freq *= lowpass_shift
    return sosfiltfilt(
        butter(4, freq / nyq, btype=band, output="sos"), data, padtype=None
    )

def ampdeamp(timeseries, ampf, amp=True):
    """
    Amplify or Deamplify timeseries.
    """
    nt = len(timeseries)
    ft_len = ampf.size + ampf.size
    ntap = int(nt * 0.05)
    timeseries[nt - ntap :] *= np.hanning(ntap * 2 + 1)[ntap + 1 :]
    timeseries = np.resize(timeseries, ft_len)
    timeseries[nt:] = 0
    fourier = rfft(timeseries)
    if not amp:
        ampf = 1.0 / ampf
    fourier[:-1] *= ampf
    return irfft(fourier)[:nt]

def transf(
    vs_soil, rho_soil, damp_soil, height_soil,
    vs_rock, rho_rock, damp_rock, nt, dt, ft_freq=None,
):
    """
    Used in deconvolution. Made by Chris de la Torre.
    """
    if ft_freq is None:
        ft_len = int(2.0 ** math.ceil(math.log(nt) / math.log(2)))
        ft_freq = np.arange(0, ft_len / 2 + 1) / (ft_len * dt)
    omega = 2.0 * math.pi * ft_freq
    Gs = rho_soil * vs_soil**2.0
    Gr = rho_rock * vs_rock**2.0
    kS = omega / (vs_soil * (1.0 + 1j * damp_soil))
    kR = omega / (vs_rock * (1.0 + 1j * damp_rock))
    alpha = Gs * kS / (Gr * kR)
    H = 2.0 / (
        (1.0 + alpha) * np.exp(1j * kS * height_soil)
        + (1.0 - alpha) * np.exp(-1j * kS * height_soil)
    )
    H[0] = 1
    return H

def read_ascii(filepath, meta=False, t0=False):
    """
    Read timeseries data from standard ascii file to numpy array.
    """
    with open(filepath, "r") as ts:
        info1 = ts.readline().split()
        info2 = ts.readline().split()
        vals = np.array(
            list(map(float, " ".join(list(map(str.rstrip, ts.readlines()))).split()))
        )
    try:
        assert len(vals) == int(info2[0])
    except AssertionError:
        print("File entries don't match NT: %s" % (filepath))
        raise
    if t0:
        diff = int(round(float(info2[4]) / float(info2[1])))
        if diff < 0:
            vals = vals[abs(diff) :]
        elif diff > 0:
            vals = np.append(np.zeros(diff), vals)
    if meta:
        note = ""
        if len(info1) >= 3:
            note = " ".join(info1[2:])
        return (
            vals,
            {
                "name": info1[0],
                "comp": info1[1],
                "note": note,
                "nt": int(info2[0]),
                "dt": float(info2[1]),
                "hr": int(float(info2[2])),
                "min": int(float(info2[3])),
                "sec": float(info2[4]),
                "e_dist": float(info2[5]),
                "az": float(info2[6]),
                "baz": float(info2[7]),
            },
        )
    return vals

def vel2acc(timeseries, dt):
    """
    Differentiate following Rob Graves' code logic.
    """
    return np.diff(np.hstack(([0], timeseries)) * (1.0 / dt))

def vel2acc3d(timeseries, dt):
    """
    vel2acc for x,y,z arrays
    """
    return np.diff(np.vstack(([0, 0, 0], timeseries)), axis=0) * (1.0 / dt)

def acc2vel(timeseries, dt):
    """
    Integrates following Rob Graves' code logic (simple).
    also works for x,y,z arrays
    """
    return np.cumsum(timeseries, axis=0) * dt

def pgv2MMI(pgv):
    """
    Calculates MMI from pgv based on Worden et al (2012)
    """
    return np.maximum(
        np.where(
            np.log10(pgv) < 0.53,
            3.78 + 1.47 * np.log10(pgv),
            2.89 + 3.16 * np.log10(pgv),
        ),
        1,
    )

def timeseries_to_text(
    timeseries: np.ndarray,
    filename: Path,
    dt: float,
    stat: str,
    comp: str,
    values_per_line: int = 6,
    start_hr: int = 0,
    start_min: int = 0,
    start_sec: float = 0.0,
    edist: float = 0.0,
    az: float = 0.0,
    baz: float = 0.0,
    title: str = "",
):
    """
    Store timeseries data into a text file.
    """
    nt = timeseries.shape[0]
    with open(filename, "wb") as txt:
        txt.write(("%-10s %3s %s\n" % (stat, comp, title)).encode())
        txt.write(
            (
                "%d %12.5e %d %d %12.5e %12.5e %12.5e %12.5e\n"
                % (nt, dt, start_hr, start_min, start_sec, edist, az, baz)
            ).encode()
        )
        divisible = nt - nt % values_per_line
        np.savetxt(
            txt, timeseries[:divisible].reshape(-1, values_per_line), fmt="%13.5e"
        )
        np.savetxt(txt, np.atleast_2d(timeseries[divisible:]), fmt="%13.5e")


# ============================================================================
# LFSeis - REVISED: Uses new timeseries_2026 parser internally
# ============================================================================
class LFSeis:
    # format constants (kept for backward compatibility)
    HEAD_STAT = 0x30
    N_COMP = 9
    # indexing constants
    X = 0
    Y = 1
    Z = 2
    COMP_NAME = {X: "090", Y: "000", Z: "ver"}

    def __init__(self, outbin):
        """
        Load LF binary store using the new 2026 parser.
        outbin: path to OutBin folder containing seis files
        """
        # Use the new parser that handles modern numpy
        self.ds = read_lfseis_directory(outbin)
        
        # Extract metadata from the new Dataset
        self.dt = self.ds.attrs["dt"]
        self.nt = len(self.ds.time)
        self.duration = self.nt * self.dt
        self.start_sec = self.ds.attrs["start_sec"]
        
        # Try to get e3d.par info if available
        self.e3dpar = os.path.join(outbin, "e3d.par")
        self.flo = None
        self.emod3d_version = None
        if os.path.isfile(self.e3dpar) or os.path.isfile(os.path.join(outbin, "../e3d.par")):
            try:
                if not os.path.isfile(self.e3dpar):
                    self.e3dpar = os.path.join(outbin, "../e3d.par")
                pars = load_e3d_par(self.e3dpar)
                self.flo = float(pars.get("flo", None))
                self.emod3d_version = pars.get("version", None)
            except Exception:
                pass  # Not critical if e3d.par can't be parsed
        
        # Build rotation matrix from dataset attributes
        rotation = self.ds.attrs.get("rotation", 0)
        theta = math.radians(rotation)
        self.rot = rotation
        self.rot_matrix = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [-math.sin(theta), -math.cos(theta), 0],
            [0, 0, -1]
        ])
        
        # Build station records matching old format
        n_stations = len(self.ds.station)
        self.stations = np.rec.array(
            np.zeros(n_stations,
                dtype=[
                    ("x", "i4"),
                    ("y", "i4"),
                    ("z", "i4"),
                    ("seis_idx", "i4", 2),
                    ("lat", "f4"),
                    ("lon", "f4"),
                    ("name", "|S8"),
                ]
            )
        )
        
        # Fill station data
        self.stations["name"] = self.ds.station.values.astype("|S8")
        self.stations["x"] = self.ds.x.values
        self.stations["y"] = self.ds.y.values
        if 'z' in self.ds:
            self.stations["z"] = self.ds.z.values
        self.stations["lat"] = self.ds.lat.values
        self.stations["lon"] = self.ds.lon.values
        
        # All stations are from a single "file" conceptually
        self.stations["seis_idx"][:, 0] = 0
        self.stations["seis_idx"][:, 1] = np.arange(n_stations)
        
        self.nstat = n_stations
        self.stat_idx = dict(zip(
            [s.decode() if isinstance(s, bytes) else s for s in self.stations["name"]],
            np.arange(self.nstat)
        ))
        
        # These are needed for some old code compatibility but no longer used
        self.seis = sorted(glob(os.path.join(outbin, "*seis-*.e3d")))
        self.ts_pos = None  # Not used with new parser
        self.ts0_type = None  # Not used
        self.ts_type = None  # Not used

    def vel(self, station, dt=None):
        """
        Returns timeseries (velocity, cm/s) for station.
        station: station name, must exist
        """
        if isinstance(station, bytes):
            station = station.decode()
        
        # Get acceleration from new parser, then integrate
        acc_data = self.ds["waveform"].sel(station=station).values  # (3, nt)
        velocity = np.cumsum(acc_data, axis=-1) * self.dt  # integrate
        
        # Return as (nt, 3) matching old API
        result = velocity.T
        
        if dt is None or dt == self.dt:
            return result
        return resample(result, int(round(self.duration / dt)))

    def acc(self, station, dt=None):
        """
        Like vel but returns acceleration (cm/s/s).
        """
        if isinstance(station, bytes):
            station = station.decode()
        
        acc_data = self.ds["waveform"].sel(station=station).values  # (3, nt)
        result = acc_data.T  # (nt, 3)
        
        if dt is None:
            dt = self.dt
        if dt == self.dt:
            return result
        return resample(result, int(round(self.duration / dt)))

    def vel2txt(self, station, prefix="./", title="", dt=None, acc=False):
        """
        Creates standard EMOD3D text files for the station.
        """
        if dt is None:
            dt = self.dt
        if acc:
            f = self.acc
        else:
            f = self.vel
        for i, c in enumerate(f(station, dt=dt).T):
            timeseries_to_text(
                c,
                f"{prefix}{station}.{self.COMP_NAME[i]}",
                dt,
                station,
                self.COMP_NAME[i],
                start_sec=self.start_sec,
                title=title,
            )

    def all2txt(self, prefix: str = "./", dt: float = None, f: str = "vel"):
        """
        Creates waveforms in text files for all stations.
        """
        if dt is None:
            dt = self.dt
        acc = f == "acc"

        prefix_chunks = prefix.split("/")
        prefix_filename = prefix_chunks[-1]
        prefix_dirname = Path("/".join(prefix_chunks[:-1])).resolve()
        prefix_dirname.mkdir(parents=True, exist_ok=True)
        for s in self.stations["name"]:
            station_name = s.decode() if isinstance(s, bytes) else s
            self.vel2txt(station_name, prefix=prefix, title=prefix_filename, dt=dt, acc=acc)


# ============================================================================
# HFSeis - PATCHED: Hardened for modern numpy
# ============================================================================
class HFSeis:
    # format constants
    HEAD_SIZE = 0x200
    HEAD_STAT = 0x18
    N_COMP = 3
    # indexing constants
    X = 0
    Y = 1
    Z = 2
    COMP_NAME = {X: "090", Y: "000", Z: "ver"}

    def __init__(self, hf_path):
        """
        Load HF binary store.
        hf_path: path to the HF binary file
        """
        hfs = os.stat(hf_path).st_size
        hff = open(hf_path, "rb")
        # determine endianness by checking file size
        nstat, nt = np.fromfile(hff, dtype="<i4", count=2)
        # Convert to Python int immediately to avoid 0-d array issues
        nstat = int(nstat)
        nt = int(nt)
        
        if (
            hfs
            == self.HEAD_SIZE
            + np.int64(nstat) * self.HEAD_STAT
            + np.int64(nstat) * nt * self.N_COMP * 4
        ):
            endian = "<"
        elif (
            hfs
            == self.HEAD_SIZE
            + np.int64(nstat) * self.HEAD_STAT  # byteswap not needed as we already converted
            + np.int64(nstat) * nt * self.N_COMP * 4
        ):
            # Try byteswapped version for big-endian
            nstat_bw = int(np.fromfile(hff, dtype=">i4", count=1)[0])
            hff.seek(4)
            nt_bw = int(np.fromfile(hff, dtype=">i4", count=1)[0])
            hff.seek(0)
            if (
                hfs
                == self.HEAD_SIZE
                + np.int64(nstat_bw) * self.HEAD_STAT
                + np.int64(nstat_bw) * nt_bw * self.N_COMP * 4
            ):
                endian = ">"
                nstat, nt = nstat_bw, nt_bw
            else:
                hff.close()
                raise ValueError("File is not an HF seis file: %s" % (hf_path))
        else:
            hff.close()
            raise ValueError("File is not an HF seis file: %s" % (hf_path))
        
        hff.seek(0)
        i4 = "%si4" % endian
        f4 = "%sf4" % endian

        # read header - integers (convert all to Python ints)
        (
            self.nstat,
            self.nt,
            self.seed,
            siteamp,
            self.pdur_model,
            nrayset,
            rayset1,
            rayset2,
            rayset3,
            rayset4,
            self.nbu,
            self.ift,
            self.nlskip,
            icflag,
            same_seed,
            site_specific_vm,
        ) = [int(x) if isinstance(x, np.ndarray) else x for x in np.fromfile(hff, dtype=i4, count=16)]
        
        self.siteamp = bool(siteamp)
        self.rayset = [rayset1, rayset2, rayset3, rayset4][:nrayset]
        self.icflag = bool(icflag)
        self.seed_inc = not bool(same_seed)
        self.site_specific_vm = bool(site_specific_vm)
        
        # read header - floats
        (
            self.duration,
            self.dt,
            self.start_sec,
            self.sdrop,
            self.kappa,
            self.qfexp,
            self.fmax,
            self.flo,
            self.fhi,
            self.rvfac,
            self.rvfac_shal,
            self.rvfac_deep,
            self.czero,
            self.calpha,
            self.mom,
            self.rupv,
            self.vs_moho,
            self.vp_sig,
            self.vsh_sig,
            self.rho_sig,
            self.qs_sig,
            self.fa_sig1,
            self.fa_sig2,
            self.rv_sig1,
        ) = np.fromfile(hff, dtype=f4, count=24)
        
        # read header - strings
        self.stoch_file, self.velocity_model = np.fromfile(hff, dtype="|S64", count=2)

        # load station info
        hff.seek(self.HEAD_SIZE)
        stations = np.fromfile(
            hff,
            count=int(self.nstat),
            dtype=[
                ("lon", f4),
                ("lat", f4),
                ("name", "|S8"),
                ("e_dist", f4),
                ("vs", f4),
            ],
        )
        hff.close()
        
        # store station names as unicode (python 3 strings)
        stat_type = stations.dtype.descr
        stat_type[2] = stat_type[2][0], "U7"
        self.stations = np.rec.fromrecords(stations, dtype=stat_type)
        if np.min(self.stations.vs) == 0:
            print("WARNING: looks like an incomplete file: %s" % (hf_path))

        # allow indexing by station names
        self.stat_idx = dict(zip(self.stations.name, np.arange(self.nstat)))
        # keep location for data retrieval
        self.path = hf_path
        # location to start of 3rd (data) block
        self.ts_pos = self.HEAD_SIZE + int(nstat) * self.HEAD_STAT
        # data format
        self.dtype = "3%sf4" % (endian)

    def acc(self, station, comp=Ellipsis, dt=None):
        """
        Returns timeseries (acceleration, cm/s/s) for station.
        station: station name, must exist
        comp: component (default all) examples: 0, self.X
        """
        with open(self.path, "r") as data:
            data.seek(self.ts_pos + self.stat_idx[station] * self.nt * 3 * 4)
            ts = np.fromfile(data, dtype=self.dtype, count=self.nt)
        if dt is None or dt == self.dt:
            return ts
        return resample(ts, int(round(self.duration / dt)))

    def vel(self, station, dt=None):
        """
        Like acc but also converts to velocity (cm/s).
        """
        if dt is None:
            dt = self.dt
        return acc2vel(self.acc(station, dt=dt), dt)

    def acc2txt(self, station, prefix="./", title="", dt=None):
        """
        Creates standard EMOD3D text files for the station.
        """
        if dt is None:
            dt = self.dt
        stat_idx = self.stat_idx[station]
        for i, c in enumerate(self.acc(station, dt=dt).T):
            timeseries_to_text(
                c,
                f"{prefix}{station}.{self.COMP_NAME[i]}",
                dt,
                station,
                self.COMP_NAME[i],
                start_sec=self.start_sec,
                edist=self.stations.e_dist[stat_idx],
                title=title,
            )

    def all2txt(self, prefix: str = "./", dt: float = None):
        """
        Creates waveforms in text files for all stations.
        """
        if dt is None:
            dt = self.dt

        prefix_chunks = prefix.split("/")
        prefix_filename = prefix_chunks[-1]
        prefix_dirname = Path("/".join(prefix_chunks[:-1])).resolve()
        prefix_dirname.mkdir(parents=True, exist_ok=True)
        for s in self.stations.name:
            self.acc2txt(s, prefix=prefix, title=prefix_filename, dt=dt)


# ============================================================================
# BBSeis - PATCHED: Hardened for modern numpy
# ============================================================================
class BBSeis:
    # format constants
    HEAD_SIZE = 0x500
    HEAD_STAT = 0x2C
    N_COMP = 3
    # indexing constants
    X = 0
    Y = 1
    Z = 2
    COMP_NAME = {X: "090", Y: "000", Z: "ver"}

    def __init__(self, bb_path):
        """
        Load BB binary store.
        bb_path: path to the BB binary file
        """
        bbs = os.stat(bb_path).st_size
        bbf = open(bb_path, "rb")
        # determine endianness by checking file size
        nstat, nt = np.fromfile(bbf, dtype="<i4", count=2)
        nstat = int(nstat)
        nt = int(nt)
        
        if (
            bbs
            == self.HEAD_SIZE
            + np.int64(nstat) * self.HEAD_STAT
            + np.int64(nstat) * nt * self.N_COMP * 4
        ):
            endian = "<"
        elif (
            bbs
            == self.HEAD_SIZE
            + np.int64(nstat) * self.HEAD_STAT
            + np.int64(nstat) * nt * self.N_COMP * 4
        ):
            nstat_bw = int(np.fromfile(bbf, dtype=">i4", count=1)[0])
            bbf.seek(4)
            nt_bw = int(np.fromfile(bbf, dtype=">i4", count=1)[0])
            bbf.seek(0)
            if (
                bbs
                == self.HEAD_SIZE
                + np.int64(nstat_bw) * self.HEAD_STAT
                + np.int64(nstat_bw) * nt_bw * self.N_COMP * 4
            ):
                endian = ">"
                nstat, nt = nstat_bw, nt_bw
            else:
                bbf.close()
                raise ValueError("File is not an BB seis file: %s" % (bb_path))
        else:
            bbf.close()
            raise ValueError("File is not an BB seis file: %s" % (bb_path))
        
        bbf.seek(0)
        i4 = "%si4" % endian
        f4 = "%sf4" % endian

        # read header - integers
        self.nstat, self.nt = [int(x) if isinstance(x, np.ndarray) else x for x in np.fromfile(bbf, dtype=i4, count=2)]
        # read header - floats
        self.duration, self.dt, self.start_sec = np.fromfile(bbf, dtype=f4, count=3)
        # read header - strings
        self.lf_dir, self.lf_vm, self.hf_file = np.fromfile(
            bbf, count=3, dtype="|S256"
        ).astype(np.unicode_)

        # load station info
        bbf.seek(self.HEAD_SIZE)
        stations = np.rec.array(
            np.fromfile(
                bbf,
                count=int(self.nstat),
                dtype=[
                    ("lon", "f4"),
                    ("lat", "f4"),
                    ("name", "|S8"),
                    ("x", "i4"),
                    ("y", "i4"),
                    ("z", "i4"),
                    ("e_dist", "f4"),
                    ("hf_vs_ref", "f4"),
                    ("lf_vs_ref", "f4"),
                    ("vsite", "f4"),
                ],
            )
        )
        bbf.close()
        
        # store station names as unicode (python 3 strings)
        stat_type = stations.dtype.descr
        stat_type[2] = stat_type[2][0], "U7"
        self.stations = np.rec.fromrecords(stations, dtype=stat_type)
        if np.min(self.stations.vsite) == 0:
            print("WARNING: looks like an incomplete file: %s" % (bb_path))

        # allow indexing by station names
        self.stat_idx = dict(list(zip(self.stations.name, np.arange(self.nstat))))
        # keep location for data retrieval
        self.path = bb_path
        # location to start of 3rd (data) block
        self.ts_pos = self.HEAD_SIZE + int(nstat) * self.HEAD_STAT
        # data format
        self.dtype = "3%sf4" % (endian)

    def acc(self, station, comp=Ellipsis):
        """
        Returns timeseries (acceleration, g) for station.
        station: station name, must exist
        comp: component (default all) examples: 0, self.X
        """
        with open(self.path, "r") as data:
            data.seek(self.ts_pos + self.stat_idx[station] * self.nt * 3 * 4)
            return np.fromfile(data, dtype=self.dtype, count=self.nt)[:, comp]

    def vel(self, station, comp=Ellipsis):
        """
        Returns timeseries (velocity, cm/s) for station.
        station: station name, must exist
        comp: component (default all) examples: 0, self.X
        """
        return acc2vel(self.acc(station, comp=comp) * 981.0, self.dt)

    def save_txt(self, station, prefix="./", title="", f="acc"):
        """
        Creates standard EMOD3D text files for the station.
        """
        stat_idx = self.stat_idx[station]
        if f == "vel":
            f = self.vel
        else:
            f = self.acc
        xyz = []
        for i, c in enumerate(f(station).T):
            xyz.append(
                timeseries_to_text(
                    c,
                    f"{prefix}{station}.{self.COMP_NAME[i]}",
                    self.dt,
                    station,
                    self.COMP_NAME[i],
                    start_sec=self.start_sec,
                    edist=self.stations.e_dist[stat_idx],
                    title=title,
                )
            )
        if prefix is None:
            return xyz

    def all2txt(self, prefix: str = "./", f: str = "acc"):
        """
        Extracts waveform data from the binary file and produces output in text format.
        """
        prefix_chunks = prefix.split("/")
        prefix_filename = prefix_chunks[-1]
        prefix_dirname = Path("/".join(prefix_chunks[:-1])).resolve()
        prefix_dirname.mkdir(parents=True, exist_ok=True)
        for s in self.stations.name:
            self.save_txt(s, prefix=prefix, title=prefix_filename, f=f)

    def save_ll(self, path):
        """
        Saves station list to text file containing: lon lat station_name.
        """
        np.savetxt(path, self.stations[["lon", "lat", "name"]], fmt="%f %f %s")


def get_observed_stations(observed_data_folder: Union[str, Path]) -> Set[str]:
    """
    returns a list of station names that can be found in the observed data folder
    """
    search_path = os.path.abspath(os.path.join(observed_data_folder, "*"))
    files = glob(search_path)
    station_names = {
        os.path.splitext(os.path.basename(filename))[0] for filename in files
    }
    return station_names

