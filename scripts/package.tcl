#
# Check configs
#
puts "**** Custom script ..."
puts "****"
puts "Number of vFPGAs: $cfg(n_reg)"

#
# Package
#
set ip_repos "$build_dir/iprepo "
for {set i 0}  {$i < $cfg(n_reg)} {incr i} {
    set cmd "ipx::infer_core -vendor user.org -library user -taxonomy /UserIP $build_dir/config_$i/ext; "
    append cmd "ipx::edit_ip_in_project -upgrade true -name config_$i -directory $build_dir/lynx/lynx.tmp $build_dir/config_$i/ext/component.xml; "
    append cmd "ipx::current_core $build_dir/config_$i/ext/component.xml; "
    append cmd "update_compile_order -fileset sources_1; "
    append cmd "set_property name top_config_$i \[ipx::current_core]; "
    append cmd "set_property display_name top_config_$i\_v1_0 \[ipx::current_core]; "
    append cmd "set_property description top_config_$i\_v1_0 \[ipx::current_core]; "
    append cmd "set_property previous_version_for_upgrade user.org:user:top:1.0 \[ipx::current_core]; "
    append cmd "set_property core_revision 1 \[ipx::current_core]; "
    append cmd "ipx::create_xgui_files \[ipx::current_core]; "
    append cmd "ipx::update_checksums \[ipx::current_core]; "
    append cmd "ipx::check_integrity \[ipx::current_core]; "
    append cmd "ipx::save_core \[ipx::current_core]; "
    append cmd "ipx::move_temp_component_back -component \[ipx::current_core]; "
    append cmd "close_project -delete; "
    eval $cmd

    append ip_repos "$build_dir/config_$i/ext "
}

set cmd "set_property  ip_repo_paths  {$ip_repos} \[current_project]; "
append cmd "update_ip_catalog; "
eval $cmd

#
# Instantiate
#

for {set i 0}  {$i < $cfg(n_reg)} {incr i} {
    create_ip -name top_config_$i -vendor user.org -library user -version 1.0 -module_name "top_config_$i\_0"
    #$generate_target all [get_ips]
    update_compile_order -fileset sources_1
}

#
# Add external sources
#
add_files $build_dir/ext

#
# Add the wrapper
#
for {set i 0}  {$i < $cfg(n_reg)} {incr i} {
    exec cp -rf "$build_dir/config_$i/user_logic_c0_$i.sv" "$proj_dir/hdl/config_0/user_logic_c0_$i.sv"
}

#
# IP cores
#
source $build_dir/init_ip.tcl -notrace

puts "**** Packaging done"
puts "****"
