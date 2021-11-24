import ketos.data_handling.database_interface as dbi 
from ketos.audio.spectrogram import MagSpectrogram
import numpy as np


# ----- create 11x_same_spec.h5 -----
data_shape = (12,12)

descr = dbi.table_description(data_shape=data_shape, include_label=False)
descr_ann = dbi.table_description_annot(freq_range=True)

f = dbi.open_file("11x_same_spec.h5", "w")

tbl = dbi.create_table(f, path='/group_1', name='table_data', description=descr)
tbl_ann = dbi.create_table(f, path='/group_1', name='table_annot', description=descr_ann)

x1 = MagSpectrogram(data=np.zeros(data_shape), time_res=1.0, freq_min=0.0, freq_res=100, filename='sine_wave')

x1.annotate(start=1., end=1.4, freq_min=50, freq_max=300, label=2)
x1.annotate(start=2., end=2.4, freq_min=60, freq_max=200, label=3)

for i in range(11):
    dbi.write(x1, tbl, table_annot=tbl_ann)

f.close()


# ----- create 15x_same_spec.h5 -----
data_shape = (12,12)

descr = dbi.table_description(data_shape=data_shape, include_label=True)

f = dbi.open_file("15x_same_spec.h5", "w")

tbl = dbi.create_table(f, path='/train', name='species1', description=descr)

x1 = MagSpectrogram(data=np.zeros(data_shape), time_res=1.0, freq_min=0.0, freq_res=100, filename='audio_file')

x1.annotate(start=1., end=1.4, freq_min=50, freq_max=300, label=1)

for i in range(15):
    dbi.write(x1, tbl)

f.close()
