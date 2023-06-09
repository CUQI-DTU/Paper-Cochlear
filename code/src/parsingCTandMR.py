import pandas as pd
import numpy as np
from os import listdir

def parse_combine_CA_ST_distances(distancefile):
    df = pd.read_csv(distancefile)
    # The first column has varying names, so we reset here to ROI.
    df = df.rename(columns={df.columns[0]: "ROI"})
    

    CA_ST_mask = ["CA" in ROIname or "ST" in ROIname for ROIname in df.iloc[:, 0]]
    df = df.loc[CA_ST_mask, :]

    CA5_loc = df.loc[df.iloc[:, 0] == "CA5", ["x microns", "y microns", "z microns"]].values[0]
    ST1_loc = df.loc[df.iloc[:, 0] == "ST1", ["x microns", "y microns", "z microns"]].values[0]
    CA5_to_ST1_dist = euclid_distance(CA5_loc, ST1_loc)

    df = df.sort_values("ROI", ascending=True)
    df.loc[5:, "distance microns"] += CA5_to_ST1_dist
    
    return df.loc[:, ["ROI", "distance microns"]]

def euclid_distance(a, b):
    return np.sqrt(sum(np.square(a-b)))

def parse_combine_CA_ST_signals(signalsfile, num_ST=5):

    CAcolumns = ["CA" + str(n) for n in range(1, 6)]
    STcolumns = ["ST" + str(n) for n in range(1, num_ST + 1)]
    usecolumns = CAcolumns + STcolumns + ["time"]

    df = pd.read_csv(signalsfile, usecols=usecolumns)
    timepoints = df.loc[:, "time"].values
    signals = df.iloc[:, 0:-2]

    signals -= np.mean(signals.iloc[:, -1])
    
    return signals, timepoints

def parse_combine_CA_ST(signalsfile, distancefile):
    signals, time = parse_combine_CA_ST_signals(signalsfile)
    distances = parse_combine_CA_ST_distances(distancefile)
    ROImask = [roi in signals.columns for roi in distances.ROI]
    distances = distances.iloc[ROImask, :]
    distances = distances.loc[:, "distance microns"].values
    
    return time, distances, signals

def loadear_CA_ST(datadir, ID):
    CT_conc_files = sorted([l for l in listdir(datadir) if "parsed" in l])
    CT_distance_files = sorted([l for l in listdir(datadir) if "distan" in l])
    ear_IDs = [fn.split("_parsed")[0] for fn in CT_conc_files]

    CT_datafiles = pd.DataFrame({"ID": ear_IDs, "conc": CT_conc_files, "dist": CT_distance_files})

    if ID in ear_IDs:
        ID_index = np.argmax(CT_datafiles.ID == ID)
    else:
        msg = f"The ear ID given ({ID}) is not amongst those found in the datadir ({datadir}):\n"
        msg += "\n".join(CT_datafiles.ID)
        raise(ValueError(msg))
    
    signalsfile = datadir + CT_datafiles.loc[ID_index, "conc"]
    distancefile = datadir + CT_datafiles.loc[ID_index, "dist"]

    time, distances, concentrations = parse_combine_CA_ST(signalsfile, distancefile)
    
    return time, distances, concentrations