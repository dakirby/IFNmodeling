from ifnclass.ifndata import IfnData, DataAlignment
import pandas as pd
import os


if __name__ == '__main__':
    newdata_1 = IfnData("20190108_pSTAT1_IFN_Bcell")
    newdata_2 = IfnData("20190119_pSTAT1_IFN_Bcell")
    newdata_3 = IfnData("20190121_pSTAT1_IFN_Bcell")
    newdata_4 = IfnData("20190214_pSTAT1_IFN_Bcell")

    alignment = DataAlignment()
    alignment.add_data([newdata_4, newdata_3, newdata_2, newdata_1])
    alignment.align()
    alignment.get_scaled_data()
    mean_data = alignment.summarize_data()
    times=mean_data.get_times()
    for s in ['Alpha', 'Beta']:
        for d in mean_data.get_doses(species=s):
            # Build file name
            file_name = s.lower() + str(int(d)) + '.exp'
            if d == 0.2:
                file_name = s.lower() + '02.exp'
            # Build data in the format required for .exp file
            data_string = "time    pSTAT    pSTAT_SD\n" # headers
            responses = mean_data.data_set.xs(s).loc[d].values
            for tidx, t in enumerate(times[s]):
                data_string += "{}    {}    {}\n".format(t, responses[tidx][0], responses[tidx][1])

            # Write to file
            with open(os.path.join(os.getcwd(), 'results', 'pybnf_data', file_name), 'w') as f:
                f.write(data_string)