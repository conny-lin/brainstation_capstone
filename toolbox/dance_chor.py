# Code from chormaster5 (June 1, 2020)

# %% JAVA ARGUMENTS----------------------------------------------------------
# % path to java programs
# javapath = fileparts(mfilename('fullpath')); % get path of this function
# b = blanks(1); % blank
# % call java 
# javacall = 'java -jar'; 
# % RAM limit
# javaRAM = '-Xmx8G'; 
# javaRAM7G = '-Xmx7G';
# % .jar paths
# beethoven = ['''',javapath,'/Beethoven_v2.jar','''']; % call beethoven 
# chor = ['''',javapath,'/Chore_1.3.0.r1035.jar','''']; % call chor 
# % chor calls 
# map = '--map';
# % settings 
# pixelsize = '-p 0.027'; 
# speed = '-s 0.1'; 
# mintime = '-t 20'; 
# nall = '-N all';
# minmove = '-M 2'; 
# shape = '--shadowless -S';
# % plugins 
# preoutline = '--plugin Reoutline::exp';  
# prespine = '--plugin Respine';

# % plugins (reversals) 
# revbeethoven_trv = '--plugin MeasureReversal::tap::dt=1::collect=0.5::postfix=trv';
# revignortap_sprevs = '--plugin MeasureReversal::postfix=sprevs';
# rev_ssr = '--plugin MeasureReversal::tap::collect=0.5::postfix=ssr';

# % dat output collection
# odrunkposture = '-O drunkposture -o nNslwakb';
# odrunkposture2 = '-O drunkposture2 -o nNslwakbcemM';
# oconny = '-O conny -o 1nee#e*ss#s*SS#S*ll#l*LL#L*ww#w*aa#a*mm#m*MM#M*kk#k*bb#b*pp#p*dd#d'; % Conny's 
# obeethoven = '-o nNss*b12M'; % standard for Beethoven
# oshanespark = '-O shanespark -o nNss*b12M'; % standard for Beethoven
# oevan = '-O evan -o nNss*b12'; % Evan's dat output
# oevanall = '-O evanall -N all -o nNss*b12';
# oswanlakeall = '-O swanlakeall -N all -o tnNemMawlkcspbd1';
# oswanlake = '-O swanlake -o tnNemMawlkcspbd1e#m#M#a#w#l#k#c#s#p#b#d#e-m-M-a-w-l-k-c-s-p-b-d-e*m*M*a*w*l*kvc*s*p*b*d*';
# onutcracker = '-O nutcracker -N all -o DfpemMwWlLaAkcsSbpdxyuvor1234';
# % Trinity app analysis for rastor plot and spontaneous locomotion
# otrinity = '-O trinity -N all -o nNss*b12xyMmeSakcr'; 
# ostarfish = '-O starfish -N all -o nNss*b12xyMmeSakcr';
# ogangnam = '-O gangnam -N all -o DpmcobdPsSruvxy1';