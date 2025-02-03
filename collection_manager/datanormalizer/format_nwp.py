import pandas as pd
import uuid
import os


class OrganizationNwp(object):
    def __init__(self, nwpflist):
        self.nwpfilelist = nwpflist

    def dremt_nwp_normal(self, outpath):
        for nwpf in self.nwpfilelist:
            nwpdf = pd.read_csv(nwpf)
            nwpdf['Date_Time']=nwpdf["Date_Time"].str.replace("_", " ")
            nwpfilename = os.path.join(outpath, nwpf.split("/")[-1][:11]+"_nwp.csv").replace("\\", "/")
            nwpdf.to_csv(nwpfilename, index=False)