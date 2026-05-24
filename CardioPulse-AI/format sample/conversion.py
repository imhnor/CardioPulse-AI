import wfdb
import csv
import json
import xml.etree.ElementTree as ET
from scipy.io import savemat

# =====================================================
# ECG RECORD NAME
# =====================================================

record_name = r"C:\Users\letsl\Documents\GitHub\ECG\extra\02000_lr"

# =====================================================
# LOAD ECG
# =====================================================

record = wfdb.rdrecord(record_name)

signals = record.p_signal
fs = record.fs
leads = record.sig_name

print("ECG Loaded Successfully")
print("Sampling Rate:", fs)
print("Leads:", leads)

# =====================================================
# SAVE CSV
# =====================================================

with open("ecg_output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(leads)
    writer.writerows(signals)

print("Saved: ecg_output.csv")

# =====================================================
# SAVE MAT
# =====================================================

savemat("ecg_output.mat", {
    "signals": signals,
    "fs": fs,
    "leads": leads
})

print("Saved: ecg_output.mat")

# =====================================================
# SAVE SCP
# =====================================================

scp_data = {
    "format": "SCP-ECG",
    "sampling_rate": fs,
    "leads": leads,
    "signals": signals.tolist()
}

with open("ecg_output.scp", "w") as f:
    json.dump(scp_data, f)

print("Saved: ecg_output.scp")

# =====================================================
# SAVE DMC
# =====================================================

dmc_data = {
    "format": "DMC",
    "sampling_rate": fs,
    "channels": leads,
    "waveform": signals.tolist()
}

with open("ecg_output.dmc", "w") as f:
    json.dump(dmc_data, f)

print("Saved: ecg_output.dmc")

# =====================================================
# SAVE XML ECG
# =====================================================

root = ET.Element("ECG")

ET.SubElement(root, "SamplingRate").text = str(fs)

leads_element = ET.SubElement(root, "Leads")

for i, lead in enumerate(leads):
    lead_element = ET.SubElement(leads_element, "Lead")
    lead_element.set("name", lead)

    signal_values = signals[:, i]

    lead_element.text = ",".join(map(str, signal_values))

tree = ET.ElementTree(root)

tree.write("ecg_output.xml")

print("Saved: ecg_output.xml")

# ======================
# ===============================
# DONE
# =====================================================

print("\nAll ECG files generated successfully.")