
# test code - doesn't work. Asking Alex Yu for sample code.
java -Xmx8G -jar "/Users/connylin/Dropbox/Code/proj/rankin_lab/Modules/Chor/Chore_1.3.0.r1035.jar" --map "/Volumes/COBOLT/MWT/20190418X_XX_100s30x10s10s_slo1/VG903_400mM/20190418_141335"

# Alex uses this
java -jar '/Users/alex/Desktop/Chore.jar' -p 0.027 -s 0.1 -t 20 -N all -M 2 --shadowless -S -o nNss*b12xyMmeSakcr --plugin Reoutline::exp --plugin Respine /Users/alex/Desktop/21_06_19_dishab_Alex/AQ2755_notap/20190621_130208.zip --map

# modified with Alex's code
java -Xmx8G -jar "/Users/connylin/Dropbox/Code/proj/rankin_lab/Modules/Chor/Chore_1.3.0.r1035.jar" -p 0.027 -s 0.1 -t 20 -N all -M 2 --shadowless -S -o nNss*b12xyMmeSakcr --plugin Reoutline::exp --plugin Respine "/Volumes/COBOLT/MWT/20190418X_XX_100s30x10s10s_slo1/VG903_400mM/20190418_141335" --map

# this one doesn't work
java -Xmx8G -jar "/Users/connylin/Dropbox/Code/proj/rankin_lab/Modules/Chor/Chore_1.3.0.r1035.jar" -p 0.027 -s 0.1 -t 20 -N all -M 2 --shadowless -S --plugin Reoutline::exp --plugin Respine "/Volumes/COBOLT/MWT/20190418X_XX_100s30x10s10s_slo1/VG903_400mM/20190418_141335" --map

# normal plate
java -Xmx8G -jar "/Users/connylin/Dropbox/Code/proj/rankin_lab/Modules/Chor/Chore_1.3.0.r1035.jar" -p 0.027 -s 0.1 -t 20 -N all -M 2 --shadowless -S -o nNss*b12xyMmeSakcr --plugin Reoutline::exp --plugin Respine '/Volumes/COBOLT/MWT/20130308C_BM_100s30x10s10s_10sISI_tolerance2/N2/20130308_090353/' --map

# etoh plate
java -Xmx8G -jar "/Users/connylin/Dropbox/Code/proj/rankin_lab/Modules/Chor/Chore_1.3.0.r1035.jar" -p 0.027 -s 0.1 -t 20 -N all -M 2 --shadowless -S -o nNss*b12xyMmeSakcr --plugin Reoutline::exp --plugin Respine '/Volumes/COBOLT/MWT/20170704X_CR_100s30x10s10s_TM5182/N2/20170704_132735/' --map

# predicted etoh but is normal (6088)
java -Xmx8G -jar "/Users/connylin/Dropbox/Code/proj/rankin_lab/Modules/Chor/Chore_1.3.0.r1035.jar" -p 0.027 -s 0.1 -t 20 -N all -M 2 --shadowless -S -o nNss*b12xyMmeSakcr --plugin Reoutline::exp --plugin Respine '/Volumes/COBOLT/MWT/20141217B_SM_100s30x10s10s/N2_400mM/20141217_121547/' --map

# wrong normal (4299)
java -Xmx8G -jar "/Users/connylin/Dropbox/Code/proj/rankin_lab/Modules/Chor/Chore_1.3.0.r1035.jar" -p 0.027 -s 0.1 -t 20 -N all -M 2 --shadowless -S -o nNss*b12xyMmeSakcr --plugin Reoutline::exp --plugin Respine '/Volumes/COBOLT/MWT/20170704X_CR_100s30x10s10s_TM5182/N2/20170704_133524/' --map
