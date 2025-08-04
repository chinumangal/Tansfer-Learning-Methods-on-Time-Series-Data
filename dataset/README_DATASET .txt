***************************************************************************************
*** Two Components of milling processes for time series prediction                  ***
***************************************************************************************

Zusammenfassung: Die Daten wurden an einer DMG CMX
600 V durch eine Siemens Industrial Edge Prozesse mit einer Abtastrate von 500 Hz
aufgenommen (mit der DOI: 10.5445/IR/1000157789). Es wurden insgesamt zwei verschiedene Bauteile aufgenommen, welche sowohl für die Bearbeitung von Stahl sowie von Aluminium verwendet wurden. Es wurden mehrere Aufnahmen mit und ohne Werkstück (Aircut) aufgenommen, um möglichst viele Fälle abdecken zu können. Zusätzlich wurde an einer DMC 60H Prozesse mit einer Abtastrate von 500 Hz durch eine Siemens Industrial Edge aufgenommen. Die Maschine wurde steuerungstechnisch aufgerüstet. Es wurden die gleichen Prozesse wie an der DMG CMX aufgenommen.Auch hier wurden mehrere Aufnahmen mit und ohne
Werkstück (Aircut) erstellt, um möglichst viele Fälle abdecken zu können. Es handelt
sich somit um die gleiche Versuchsreihe wie in "Training and validation dataset of milling
processes for time series prediction" mit der DOI 10.5445/IR/1000157789. Die aufgenommenen Daten wurde danach dem Bauteil und Material entsprechend in einem Datenframe zusammengeführt, geglättet und auf 50Hz downgesampelt. 
Die simulierten Kräfte der DMC 60H wurden durch einen experimentell ermittelten Maschinenkoeffizienten dividiert, um eine präzisere Approximation der realen Kräfte zu erzielen. Die Koeffizienten betragen 5,42189 für Aluminium und 2,78245 für Stahl.

Abstract: The data was measured on a DMG CMX
600 V by a Siemens Industrial Edge processes with a sampling rate of 500 Hz.
were recorded. A total of two different components were recorded, which were used for machining both steel and aluminum. Several recordings were made with and without the workpiece (air cut) in order to cover as many cases as possible. In addition, processes were recorded on a DMC 60H with a sampling rate of 500 Hz using a Siemens Industrial Edge. The machine was upgraded in terms of control technology. The same processes were recorded as on the DMG CMX.
workpiece (air cut) in order to cover as many cases as possible. It is therefore
The test series is therefore the same as in "Training and validation dataset of milling
processes for time series prediction" with the DOI 10.5445/IR/1000157789. The recorded data was then merged into a data frame according to the component and material, smoothed and downsampled to 50Hz.
The simulated forces of the DMC 60H were divided by an experimentally determined machine coefficient to achieve a more precise approximation of the real forces. The coefficients are 5.42189 for aluminum and 2.78245 for steel. 

---------------------------------------------------------------------------------------

Documents:
-Design of Experiments: Information on the paths such as the technological values of
 the experiments
-Recording information: Information about the recordings with comments
-Data: All dataframes splittet by milling machine, material, component and process.
-NC-Code: NC programs executed on the machine


Experimental data:
-Machine: DMG CMX 600 V
-Material: S235JR, 2007 T4
-Tools:
   -VHM-Fräser HPC, TiSi, ⌀ f8 DC: 5mm
   -VHM-Fräser HPC, TiSi, ⌀ f8 DC: 10mm
   -VHM-Fräser HPC, TiSi, ⌀ f8 DC: 20mm
   -Schaftfräser HSS-Co8, TiAlN, ⌀ k10 DC: 5mm
   -Schaftfräser HSS-Co8, TiAlN, ⌀ k10 DC: 10mm
   -Schaftfräser HSS-Co8, TiAlN, ⌀ k10 DC: 5mm
-Workpiece blank dimensions: 150x75x50mm

License: This work is licensed under a Creative Commons Attribution 4.0 International
License. Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0).


Experimental data:
-Machine: Retrofitted DMC 60H
-Material: S235JR, 2007 T4
-Tools:
   -VHM-Fräser HPC, TiSi, ⌀ f8 DC: 5mm
   -VHM-Fräser HPC, TiSi, ⌀ f8 DC: 10mm
   -VHM-Fräser HPC, TiSi, ⌀ f8 DC: 20mm
   -Schaftfräser HSS-Co8, TiAlN, ⌀ k10 DC: 5mm
   -Schaftfräser HSS-Co8, TiAlN, ⌀ k10 DC: 10mm
   -Schaftfräser HSS-Co8, TiAlN, ⌀ k10 DC: 5mm
-Workpiece blank dimensions: 150x75x50mm

License: This work is licensed under a Creative Commons Attribution 4.0 International
License. Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0).
