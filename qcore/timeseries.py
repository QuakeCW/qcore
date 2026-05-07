"""
Shared functions to work on time-series.

@author Viktor Polak
@date 13/09/2016

Patched 2026: LFSeis uses new parser for metadata, direct file reads for waveforms.
               HFSeis/BBSeis hardened for numpy 1.x+
"""

import base64
from glob import glob
from io import BytesIO, BufferedReader
import math
import os
import pandas as pd
from pathlib import Path
from subprocess import Popen, PIPE
from typing import NamedTuple, Union, Set

from qcore.constants import MAXIMUM_EMOD3D_TIMESHIFT_1_VERSION
from qcore.formats import load_e3d_par
from qcore.utils import compare_versions

# Import from new parser - only use LFSeisParser for metadata, not read_lfseis_directory
#from qcore.timeseries_2026 import LFSeisParser

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

def seis2txt(
    seis,
    dt,
    prefix,
    stat,
    comp,
    start_hr=0,
    start_min=0,
    start_sec=0.0,
    edist=0.0,
    az=0.0,
    baz=0.0,
    title="",
    vpl=6,
):
    """
    Store timeseries data as standard EMOD3D text file {prefix}{stat}.{comp}.
    seis: timeseries
    dt: timestep
    prefix: filename excluding station name and extention, None to return contents as byte array.
    stat: station name
    comp: same as file extention ('090', '000', 'ver')
    start_hr: start time (hours, generally not used)
    start_min: start time (minutes, generally not used)
    start_sec: start time (seconds)
    edist: epicentre distance
    az: generally not used
    baz: generally not used
    title: used in header
    vpl: values per line, more is faster
    """
    nt = seis.shape[0]
    if prefix is None:
        txt = BytesIO()
    else:
        txt = open("%s%s.%s" % (prefix, stat, comp), "wb")

    # same format strings as fdbin2wcc
    txt.write(("%-10s %3s %s\n" % (stat, comp, title)).encode())
    txt.write(
        (
            "%d %12.5e %d %d %12.5e %12.5e %12.5e %12.5e\n"
            % (nt, dt, start_hr, start_min, start_sec, edist, az, baz)
        ).encode()
    )
    # values below header lines, vpl per line
    divisible = nt - nt % vpl
    np.savetxt(txt, seis[:divisible].reshape(-1, vpl), fmt="%13.5e")
    np.savetxt(txt, np.atleast_2d(seis[divisible:]), fmt="%13.5e")

    if prefix is None:
        return txt.getvalue()
    else:
        txt.close()

_HEAD_STAT = 48  # Header size per station
_N_COMP = 9  # Number of components in LF seis files

class LFSeisHeader(NamedTuple):
    """LFSeis File Header."""

    nstat: int
    """int: The number of stations in the file."""
    nt: int
    """int: The number of timesteps the file."""
    dt: float
    """float: The temporal resolution of the file."""
    resolution: float
    """float: The spatial resolution of the file."""
    rotation: float
    """float: The model rotation."""


class LFSeisParser:
    """A parser for LFSeis files.

    Parameters
    ----------
    handle : BufferedReader
        A buffered reader object representing a file object. Cannot be a
        file-like object due to the use of `np.fromfile` which requires a
        `fileno`.
    """

    def __init__(self, handle: BufferedReader):  # noqa: D107
        self.handle = handle
        self.length = self._extract_length()
        self.i4, self.f4 = self._lfseis_dtypes()

    def _extract_length(self) -> int:
        """Extract the length of the file, in bytes.

        Returns
        -------
        int
            The file length.
        """
        current_position = self.handle.tell()
        self.handle.seek(0, os.SEEK_END)
        length = self.handle.tell()
        self.handle.seek(current_position)
        return length

    def _lfseis_dtypes(self) -> tuple[str, str]:
        """Heuristically evaluate the byte order of the file.



        Returns
        -------
        tuple[str, str]
            The numpy dtype strings representing the `self.i4` and
            `self.f4` types accounting for machine byteorder.

        Raises
        ------
        ValueError
            If the given file does not have an `nt`, `nstat` value
            consistent with the file size. Will try byteorder swapping
            the values to find a match.
        """
        current_position = self.handle.tell()
        nstat_raw = np.fromfile(self.handle, dtype="<i4", count=1)[0]
        _ = self.handle.seek(
            16, os.SEEK_CUR
        )  # 4 * i4 for station index, x, y, z grid point of the first station
        nt_raw = np.fromfile(self.handle, dtype="<i4", count=1)[0]
        nstat = int(nstat_raw)
        nt = int(nt_raw)
        nstat_bw = int(nstat_raw.byteswap())
        nt_bw = int(nt_raw.byteswap())
        if self.length == 4 + nstat * _HEAD_STAT + nstat * nt * _N_COMP * 4:
            endian = "<"
        elif self.length == 4 + nstat_bw * _HEAD_STAT + nstat_bw * nt_bw * _N_COMP * 4:
            endian = ">"
        else:
            raise ValueError("Handle does not read from an LFSeis file.")

        i4 = f"{endian}i4"
        f4 = f"{endian}f4"
        self.handle.seek(current_position)
        return i4, f4

    def read_header(self) -> LFSeisHeader:
        """Read the header values from an LFSeis file.

        Returns
        -------
        LFSeisHeader
            The header object representing this file.
        """
        nstat = int(np.fromfile(self.handle, dtype=self.i4, count=1)[0])
        current_position = self.handle.tell()
        _ = self.handle.seek(
            16, os.SEEK_CUR
        )  # 4 * i4 for station index, x, y, z grid point of the first station
        nt = int(np.fromfile(self.handle, dtype=self.i4, count=1)[0])
        dt, resolution, rotation = map(
            float, np.fromfile(self.handle, dtype=self.f4, count=3)
        )
        self.handle.seek(current_position)
        return LFSeisHeader(
            nstat=nstat, nt=nt, dt=dt, resolution=resolution, rotation=rotation
        )

    def read_stations(self, nstat: int) -> pd.DataFrame:
        """Read the stations table from the LFSeis file.

        Parameters
        ----------
        nstat : int
            The number of stations to read.

        Returns
        -------
        pd.DataFrame
            A dataframe of all the stations contained in this file.
        """
        dtype_header = np.dtype(
            {
                "names": [
                    "stat_pos",
                    "x",
                    "y",
                    "lat",
                    "lon",
                    "station",
                ],
                "formats": [
                    self.i4,
                    self.i4,
                    self.i4,
                    self.f4,
                    self.f4,
                    "|S8",
                ],
                # Stations are packed like:
                # x y z seis_idx nt timesteps resolution rotation lat lon name
                # with timesteps, resolution and rotation repeated for
                # each station. So we skip these values using the
                # offsets.
                "offsets": [0, 4, 8, 32, 36, 40],
            }
        )
        station_headers = np.fromfile(self.handle, dtype=dtype_header, count=nstat)
        x_coords: npt.NDArray[np.int32] = station_headers["x"]
        y_coords: npt.NDArray[np.int32] = station_headers["y"]
        lat_coords: npt.NDArray[np.int32] = station_headers["lat"]
        lon_coords: npt.NDArray[np.int32] = station_headers["lon"]
        # Decode station names from UTF-8 encoded bytes.
        station_names: list[str] = [
            station_name.decode("utf-8", errors="replace").strip("\x00")
            for station_name in station_headers["station"]
        ]
        return pd.DataFrame(
            {
                "x": x_coords,
                "y": y_coords,
                "lat": lat_coords,
                "lon": lon_coords,
                "station": station_names,
            }
        )

    def read_waveform(self, shape: tuple[int, ...]) -> np.memmap:
        """Memory map the waveform array from the seis file.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the resulting waveform array.

        Returns
        -------
        np.memmap
            A memory-mapped array containing floating point values
            read from the seis file in the provided shape.
        """
        return np.memmap(
            self.handle, dtype=self.f4, mode="r", offset=self.handle.tell(), shape=shape
        )



# ============================================================================
# LFSeis - REVISED: Uses new timeseries_2026 parser, memory-efficient
# ============================================================================
class LFSeis:
    HEAD_STAT = 0x30
    N_COMP = 9
    X = 0
    Y = 1
    Z = 2
    COMP_NAME = {X: "090", Y: "000", Z: "ver"}

    def __init__(self, outbin):
        """Load LF binary store using 2026 parser for metadata + direct file reads."""
        self.outbin = Path(outbin)
        self.seis = sorted(glob(os.path.join(outbin, "*seis-*.e3d")))
        
        if not self.seis:
            raise ValueError(f"No LF seis files found in {outbin}")
        
        # Use new parser for metadata only
        header_file = next(s for s in self.seis if os.stat(s).st_size > 0)
        with open(header_file, "rb") as f:
            parser = LFSeisParser(f)
            header = parser.read_header()
        
        self.dt = header.dt
        self.nt = header.nt
        self.duration = self.nt * self.dt
        self.resolution = header.resolution
        self.rot = header.rotation
        
        # Rotation matrix
        theta = math.radians(self.rot)
        self.rot_matrix = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [-math.sin(theta), -math.cos(theta), 0],
            [0, 0, -1]
        ])
        
        # Build station index and file offset map
        self._station_offset = {}
        all_stations = []
        
        for seis_file in self.seis:
            with open(seis_file, "rb") as f:
                parser = LFSeisParser(f)
                f_header = parser.read_header()
                stations_df = parser.read_stations(f_header.nstat)
                
                data_start = 4 + f_header.nstat * self.HEAD_STAT
                
                for i, row in stations_df.iterrows():
                    name = row["station"]
                    if isinstance(name, bytes):
                        name = name.decode('utf-8').strip('\x00').strip()
                    else:
                        name = name.strip()
                    
                    offset = data_start + i * f_header.nt * self.N_COMP * 4
                    self._station_offset[name] = (seis_file, offset, f_header.nt)
                    all_stations.append({
                        "name": name,
                        "x": row["x"],
                        "y": row["y"],
                        "lat": row["lat"],
                        "lon": row["lon"],
                    })
        
        # Build stations array in the order we read them (HF will match by name, not position)
        n_stations = len(all_stations)
        self.stations = np.rec.array(
            np.zeros(n_stations,
                dtype=[
                    ("x", "i4"),
                    ("y", "i4"),
                    ("z", "i4"),
                    ("seis_idx", "i4", 2),
                    ("lat", "f4"),
                    ("lon", "f4"),
                    ("name", "U7"),
                ]
            )
        )
        
        for i, s in enumerate(all_stations):
            self.stations[i]["name"] = s["name"]
            self.stations[i]["x"] = s["x"]
            self.stations[i]["y"] = s["y"]
            self.stations[i]["z"] = 0
            self.stations[i]["lat"] = s["lat"]
            self.stations[i]["lon"] = s["lon"]
            self.stations[i]["seis_idx"][0] = 0
            self.stations[i]["seis_idx"][1] = i
        
        self.nstat = n_stations
        self.stat_idx = {s["name"]: i for i, s in enumerate(all_stations)}
        
        # e3d.par
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
                pass
        
        if self.flo is not None:
            self.start_sec = -1 / self.flo
            if self.emod3d_version is not None and compare_versions(
                self.emod3d_version, MAXIMUM_EMOD3D_TIMESHIFT_1_VERSION
            ) > 0:
                self.start_sec *= 3
        else:
            self.start_sec = -3 * (0.5 / (5 * self.resolution))
        
        self.ts_pos = None
        self.ts0_type = None
        self.ts_type = None
    
    def _read_station_waveform(self, station):
        """Read a single station's waveform from .e3d file."""
        if isinstance(station, bytes):
            station = station.decode()
        
        filepath, offset, nt = self._station_offset[station]
        
        if not hasattr(self, '_endian'):
            with open(self.seis[0], "rb") as f:
                parser = LFSeisParser(f)
                self._endian = parser.f4[0]
                self._f4 = parser.f4
        
        with open(filepath, "rb") as f:
            f.seek(offset)
            raw = np.fromfile(f, dtype=self._f4, count=nt * self.N_COMP)
            raw = raw.reshape(nt, self.N_COMP)
            xyz = raw[:, :3]
            rotated = np.dot(xyz, self.rot_matrix)
            acc = np.gradient(rotated, self.dt, axis=0)
            return acc

    def vel(self, station, dt=None):
        acc_data = self._read_station_waveform(station)
        velocity = np.cumsum(acc_data, axis=0) * self.dt
        if dt is None or dt == self.dt:
            return velocity
        return resample(velocity, int(round(self.duration / dt)))

    def acc(self, station, dt=None):
        acc_data = self._read_station_waveform(station)
        if dt is None:
            dt = self.dt
        if dt == self.dt:
            return acc_data
        return resample(acc_data, int(round(self.duration / dt)))

    def vel2txt(self, station, prefix="./", title="", dt=None, acc=False):
        if dt is None:
            dt = self.dt
        f = self.acc if acc else self.vel
        for i, c in enumerate(f(station, dt=dt).T):
            timeseries_to_text(
                c, f"{prefix}{station}.{self.COMP_NAME[i]}", dt,
                station, self.COMP_NAME[i], start_sec=self.start_sec, title=title,
            )

    def all2txt(self, prefix="./", dt=None, f="vel"):
        if dt is None:
            dt = self.dt
        acc = f == "acc"
        prefix_chunks = prefix.split("/")
        prefix_filename = prefix_chunks[-1]
        prefix_dirname = Path("/".join(prefix_chunks[:-1])).resolve()
        prefix_dirname.mkdir(parents=True, exist_ok=True)
        for s in self.stations["name"]:
            self.vel2txt(s, prefix=prefix, title=prefix_filename, dt=dt, acc=acc)


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
            + np.int64(nstat) * self.HEAD_STAT
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
        ).astype(np.str_)

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
