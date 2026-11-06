# Vivado batch implementation of this core only. No board or pinout implied.
set part $::env(PART)
set period $::env(CLK_NS)
create_project -in_memory -part $part
read_verilog -sv rtl/selective_scan.sv
synth_design -top selective_scan -part $part
create_clock -name core_clk -period $period [get_ports clk]
report_utilization -file build/vivado/utilization_synth.rpt
opt_design
place_design
route_design
report_timing_summary -file build/vivado/timing_routed.rpt
report_utilization -file build/vivado/utilization_routed.rpt
write_checkpoint -force build/vivado/selective_scan_routed.dcp
puts "FINISHED: inspect utilization_routed.rpt and timing_routed.rpt"
