#!/usr/bin/env python3
"""
create db:
create_imdb(runs_dir, station_file, db_file)
or run as __main__ --help

load db at station (pandas dataframe):
station_ims(imdb_file, station, im=default_all)

find closest station:
closest_station(imdb_file, lon, lat)
"""

from glob import glob
from multiprocessing import Pool
import os
import sqlite3

import numpy as np
import pandas as pd


def __init_db(conn, ims):
    c = conn.cursor()

    c.execute(
        """CREATE TABLE `ims` (
                    `im_name` TEXT NOT NULL UNIQUE
                 );"""
    )
    c.executemany(
        """INSERT INTO `ims`(`im_name`)
                        VALUES (?)""",
        [[im] for im in ims],
    )

    c.execute(
        """CREATE TABLE `stations` (
                    `id` INTEGER PRIMARY KEY AUTOINCREMENT,
                    `name` VARCHAR(8) NOT NULL UNIQUE,
                    `longitude` FLOAT NOT NULL,
                    `latitude` FLOAT NOT NULL
                 );"""
    )

    c.execute(
        """CREATE TABLE `simulations` (
                    `id` INTEGER PRIMARY KEY AUTOINCREMENT,
                    `name` TEXT NOT NULL UNIQUE
                 );"""
    )

    for im in ims:
        c.execute(
            """CREATE TABLE `%s` (
                        `station_id` INTEGER,
                        `simulation_id` INTEGER,
                        `value` FLOAT NOT NULL,
                        FOREIGN KEY(`station_id`) REFERENCES `stations`(`id`),
                        FOREIGN KEY(`simulation_id`) REFERENCES `simulation`(`id`)
                     );"""
            % (im)
        )

    conn.commit()


def __add_simulation(conn, simulation_name):
    """
    Add a new simulation into the database.
    conn: open connection to database
    simulation_name: name of the simulation to add
    return: id of the simulation added
    """
    c = conn.cursor()
    c.execute("""INSERT INTO `simulations`(`name`) VALUES (?)""", (simulation_name,))
    conn.commit()

    return c.lastrowid


def __expand_stations(conn, station_ids, stations, station_ll):
    """
    Add stations that aren't alredy in database, update station_ids dict.
    conn: open connection to database
    station_ids: dictionary of known station_name -> station_id
    stations: potentially new station_name(s)
    station_ll: loction dict for stations
    """
    stations_new = [s for s in stations if s not in station_ids]
    if len(stations_new) == 0:
        return

    c = conn.cursor()
    for s in stations_new:
        c.execute(
            """INSERT INTO `stations`(`name`, `longitude`, `latitude`)
                        VALUES (?,?,?)""",
            (s, station_ll[s][0], station_ll[s][1]),
        )
        station_ids[s] = c.lastrowid
    conn.commit()


def __store_ims(conn, table, station_ids, sim_id):
    """
    Store IMs in SQLite file from individual simulation CSV files.
    conn: open connection to database
    table: pandas dataframe containing data
    station_ids: dictionary of station_name -> station_id
    sim_id: simulation_id
    """
    c = conn.cursor()
    stations = [station_ids[s] for s in table.index.values]
    n_stat = len(stations)
    simulations = [sim_id] * n_stat

    for im in table.columns.values:
        values = zip(stations, simulations, table[im].values)
        c.executemany(
            """INSERT INTO `%s`(`station_id`, `simulation_id`, `value`)
                            VALUES (?, ?, ?)"""
            % (im),
            values,
        )
    conn.commit()


def __pandas_loader(csv):
    """
    Loads individual simulation CSV files in background.
    """
    c = pd.read_csv(csv, index_col=0)
    return c.loc[c["component"] == "geom"].drop("component", axis="columns")


def __im_at_station(imdb_file, im, station_id, n_sim):
    """
    SQLite retrieval of IMs. Run internally with multiprocessing.Pool.map.
    imdb_file: location of database
    im: im to load
    station_id: station of interest
    n_sim: how many values are expected
    """
    station_values = np.zeros((2, n_sim))
    conn = sqlite3.connect(imdb_file)
    c = conn.cursor()
    c.execute(
        """SELECT `simulation_id`, `value` FROM `%s`
                    WHERE `station_id` = %d
                    ORDER BY `simulation_id`"""
        % (im, station_id)
    )
    for r, row in enumerate(c):
        station_values[:, r] = row
    conn.close()
    return station_values


def __im_at_station_star(imdbfile_im_stationid_nsim):
    """
    Python 2/3 compatable version for multi-argument map.
    """
    return __im_at_station(*imdbfile_im_stationid_nsim)


def create_imdb(runs_dir, station_file, db_file, nproc=1):
    """
    Create SQLite database for IMs across simulations.
    runs_dir: location to `Runs` folder in a Cybershake run
    db_file: where to store output
    nproc: number of processes to use (not very useful anyway)
    """
    if os.path.exists(db_file):
        # no support for updating
        os.remove(db_file)

    # create imdb for potentially a subset of all stations in IM CSVs
    # station list needed in all cases as location is extracted from here
    station_ll = {}
    with open(station_file, "r") as sf:
        for line in sf:
            lon, lat, name = line.split()
            station_ll[name] = (lon, lat)

    def listed_stations(stations):
        """
        Returns subset of stations that are in given station list file.
        """
        return [s for s in stations if s in station_ll]

    csvs = glob(os.path.join(runs_dir, "*", "IM_calc", "*", "*.csv"))
    sims = list(map(lambda path: os.path.splitext(os.path.basename(path))[0], csvs))
    faults = list(map(lambda sim: sim.split("_HYP")[0], sims))

    pool = Pool(nproc)
    sink = {}
    step = min(nproc, len(csvs))
    for i in range(step):
        sink[i] = pool.apply_async(__pandas_loader, (csvs[i],))

    conn = sqlite3.connect(db_file)
    __init_db(conn, sink[0].get().columns.values.tolist())
    station_ids = {}

    for i, csv in enumerate(csvs):
        # keep background csv load processes busy
        c = sink[i].get()
        del sink[i]
        try:
            sink[i + step] = pool.apply_async(__pandas_loader, (csvs[i + step],))
        except IndexError:
            pass

        # don't bother if no stations
        interesting_stations = listed_stations(c.index.values)
        if len(interesting_stations) == 0:
            print("Simulation contains no stations of interest: %s" % (sims[i]))
            continue

        # save in DB
        sim_id = __add_simulation(conn, sims[i])
        __expand_stations(conn, station_ids, interesting_stations, station_ll)
        __store_ims(conn, c.loc[interesting_stations], station_ids, sim_id)
        print("CSV %d of %d..." % (sim_id, len(csvs)))
    print("CSV loading complete.")

    conn.close()


# FUNCTIONS BELOW HERE COULD BE COMBINED INTO A CLASS


def ims(imdb_file, fmt="imdb"):
    """
    Returns list of IMs available in IMDB
    """

    conn = sqlite3.connect(imdb_file)
    c = conn.cursor()
    c.execute("""SELECT `im_name` FROM `ims`""")
    ims = [row[0] for row in c]
    conn.close()

    if fmt == "file":
        fmt_file = (
            lambda im: im if not im.startswith("pSA") else im[1:].replace(".", "p")
        )
        return list(map(fmt_file, ims))

    return ims


def station_ims(imdb_file, station, im=None, nproc=None):
    """
    Load IMs for a given station.
    station: load IMs for this station
    im: only give this IM
    nproc: limit loading processes (default nproc=n_im)
    """

    csv = os.path.join("%s_stations" % (imdb_file), "%s.csv" % (station))
    if os.path.exists(csv):
        dataframe = pd.read_csv(csv, index_col=0)
        if im is not None:
            return dataframe[im]
        return dataframe

    conn = sqlite3.connect(imdb_file)
    c = conn.cursor()

    # station ID and IMs in DB
    c.execute("""SELECT `id` FROM `stations` WHERE `name` = (?)""", (station,))
    try:
        station_id = c.fetchall()[0][0]
    except IndexError:
        print("station not found in IM DB: %s" % (station))
        conn.close()
        return
    c.execute("""SELECT `im_name` FROM `ims`""")
    im_names = [row[0] for row in c]
    n_im = len(im_names)

    # determine the number of simulations
    c.execute(
        """SELECT COUNT(*) FROM `%s`
                    WHERE `station_id` = %d"""
        % (im_names[0], station_id)
    )
    n_sim = c.fetchall()[0][0]

    # gather IMs
    if nproc is None:
        nproc = n_im
    pool = Pool(min(nproc, n_im))
    im_values = np.zeros((n_sim, n_im))
    id_vals = pool.map(
        __im_at_station_star,
        zip([imdb_file] * n_im, im_names, [station_id] * n_im, [n_sim] * n_im),
    )
    simulation_ids = id_vals[0][0]
    for i in range(len(id_vals)):
        im_values[:, i] = id_vals[i][1]
    del id_vals

    c.execute(
        """SELECT `name` FROM `simulations`
                    WHERE `id` in (%s)
                    ORDER BY `id`"""
        % (",".join(map(str, simulation_ids)))
    )
    gm_names = [row[0] for row in c]

    conn.close()

    df = pd.DataFrame(im_values, index=gm_names, columns=im_names)
    if not os.path.isdir(os.path.dirname(csv)):
        os.makedirs(os.path.dirname(csv))
    df.to_csv(csv)
    if im is not None:
        return df[im]
    return df


def closest_station(imdb_file, lon, lat):
    """
    Find closest station.
    imdb_file: SQLite database file
    lon: target longitude
    lat: target latitude
    returns: numpy.record with fields: id, name, lon, lat, dist
    """
    conn = sqlite3.connect(imdb_file)
    c = conn.cursor()
    c.execute("""SELECT `id`,`name`,`longitude`,`latitude`,`id` FROM `stations`""")
    r = np.rec.array(
        np.array(
            c.fetchall(),
            dtype={
                "names": ["id", "name", "lon", "lat", "dist"],
                "formats": ["i4", "S7", "f4", "f4", "f4"],
            },
        )
    )
    conn.close()

    d = (
        np.sin(np.radians(r.lat - lat) / 2.0) ** 2
        + np.cos(np.radians(lat))
        * np.cos(np.radians(r.lat))
        * np.sin(np.radians(r.lon - lon) / 2.0) ** 2
    )
    r.dist = 6378.139 * 2.0 * np.arctan2(np.sqrt(d), np.sqrt(1 - d))

    return r[np.argmin(r.dist)]


def station_details(imdb_file, station_name=None, station_id=None):
    """
    Give station details given name or id. Return all stations if no selection.
    """
    conn = sqlite3.connect(imdb_file)
    c = conn.cursor()
    if station_name is not None:
        c.execute(
            """SELECT `id`,`name`,`longitude`,`latitude` FROM `stations`
                     WHERE `name` = (?)""",
            (station_name,),
        )
    elif station_id is not None:
        c.execute(
            """SELECT `id`,`name`,`longitude`,`latitude` FROM `stations`
                     WHERE `id` = (?)""",
            (station_id,),
        )
    else:
        c.execute("""SELECT `id`,`name`,`longitude`,`latitude` FROM `stations`""")
    r = np.rec.array(
        np.array(
            c.fetchall(),
            dtype={
                "names": ["id", "name", "lon", "lat"],
                "formats": ["i4", "S7", "f4", "f4"],
            },
        )
    )
    conn.close()

    if r.size == 1:
        # specific station_name or station_id, both are unique
        return r[0]
    # list of all stations
    return r


def fill_cache(imdb_file):
    """
    Makes sure cache files are available for every station.
    """
    for station in station_details(imdb_file):
        station_ims(imdb_file, station.name)


if __name__ == "__main__":
    """
    Command line option to save database.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    arg = parser.add_argument
    arg("runs_dir", help="Location of Runs folder")
    arg("station_file", help="Location of station (ll) file")
    arg("db_file", help="Where to store IMDB")
    arg("--cache", help="Create cache files at every station too", action="store_true")
    arg("--nproc", help="Number of processes to use", type=int, default=1)
    args = parser.parse_args()

    create_imdb(args.runs_dir, args.station_file, args.db_file, nproc=args.nproc)
    if args.cache:
        fill_cache(args.db_file)