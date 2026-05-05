
# Snapshot Analysis

All episodes are represented by [pid, episode number].
MS = Max Steps Case
FS = Fold Stall Case
WB = Wrong Branch Case

## 1. RCCA/RVA:- 
### a. MS: In MS cases Passing all entries in the main aorta (going all the way to the -ve x side) and then curving onto the end of main aorta mesh and having high negative reward < -60 (ex: [312, 1], [502, 1], [122/578/616, 1]) which is worse than MS cases when the wire is stuck just before the entry point of LVA (and the target is in RCCA and RVA; similar pattern for LCCA targets too; ex: [274/160/198, 2], [388, 2]). In some MS cases it's better and about the same reward as when stuck before LVA entry point (reward < -30 but > -35; ex: [160/388/540, 3], [350/464/502, 3]). This should not be case as wire went way further but crossed the entry points of RCCA/RVA. Also why is wire travelling that far since as soon as it passes the entry point of RCCA/RVA it should be off_path and in retract mode with a newer planned path (different from initialized one).

### b. FS: FS only happens in the RCCA cases [160/72/426, 1] and they happend in the same end orientation as MS; again why isn't wire retracting way sooner and what's the different episodes evolution that's causing thgis difference


## 2 LCCA:-
### a. MS: Stuck right outside LCCA entry episode got the worst reward (-54; [350, 1]) as compared to suck right before LVA/RVA/RCCA entry episodes (-30s, [122/502/578, 2]) . Analyse and Debug

### b. FS: Catheter seems stuck at/near LCCA entry but guidewire is hanging out in the middle of the vessel (aorta); what's the problem ; why can't it move forward / retract 

## 3. LVA:
### a. FS. I think it does make it to LVA entry but gets stuck , why ? too much rotation ?

### b. MS: stuck at/near LCCA entry
