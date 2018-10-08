def get_kfactors (results):
        
    kf_Al_K = 1.05
    kf_B_K = 5
    kf_C_K = 0.95
    kf_Ca_K = 1.16
    kf_Ca_L = 9.6
    kf_Cl_K = 1.093
    kf_Cr_K = 1.314
    kf_Cr_L = 5.23
    kf_Cu_K = 1.773
    kf_Cu_L = 1.523
    kf_F_K = 1
    kf_Fe_K = 1.51
    kf_Fe_L = 2.35
    kf_Ga_K = 1
    kf_Ga_L = 1
    kf_K_K = 1.119
    kf_Mg_K = 0.94
    kf_Mn_K = 1.5
    kf_Mn_L = 3.36
    kf_N_K = 1
    kf_Na_K = 0.88
    kf_Ni_K = 1.587
    kf_Ni_L = 1.582
    kf_O_K = 0.94
    kf_P_K = 1.096
    kf_Pt_L = 1
    kf_Pt_M = 1
    kf_S_K = 1.54
    kf_Si_K = 1
    kf_Ti_K = 1.25
    kf_Zr_K = 3.65
    kf_Zr_L = 1.49
    
    kfactors=[]
    
    for line in range (0, len(results)):
        if "Ag_La" in results[line].metadata.Sample.xray_lines: 
            kfactors.append(1)
        elif "Al_Ka" in results[line].metadata.Sample.xray_lines: 
            kfactors.append(kf_Al_K)
        elif "B_Ka" in results[line].metadata.Sample.xray_lines: 
            kfactors.append(kf_B_K)
        elif "Bi_Ma" in results[line].metadata.Sample.xray_lines: 
            kfactors.append(1)
        elif "C_Ka" in results[line].metadata.Sample.xray_lines: 
            kfactors.append(kf_C_K)
        elif "Ca_Ka" in results[line].metadata.Sample.xray_lines: 
            kfactors.append(kf_Ca_K)
        elif "Ca_La" in results[line].metadata.Sample.xray_lines: 
            kfactors.append(kf_Ca_L)
        elif "Cl_Ka" in results[line].metadata.Sample.xray_lines: 
            kfactors.append(kf_Cl_K)
        elif "Cr_Ka" in results[line].metadata.Sample.xray_lines: 
            kfactors.append(kf_Cr_K)        
        elif "Cr_La" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Cr_L)
        elif "Cu_Ka" in results[line].metadata.Sample.xray_lines: 
            kfactors.append(kf_Cu_K)
        elif "Cu_La" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Cu_L)
        elif "F_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_F_K)        
        elif "Fe_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Fe_K)
        elif "Fe_La" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Fe_L)
        elif "Fe_Lb3" in results[line].metadata.Sample.xray_lines:
            kfactors.append(0)
        elif "Fe_Ll" in results[line].metadata.Sample.xray_lines:
            kfactors.append(0)
        elif "Fe_Ln" in results[line].metadata.Sample.xray_lines:
            kfactors.append(0)
        elif "Ga_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Ga_K)             
        elif "Ga_La" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Ga_L)
        elif "Ho_La" in results[line].metadata.Sample.xray_lines: 
            kfactors.append(1)
        elif "La_La" in results[line].metadata.Sample.xray_lines: 
            kfactors.append(1)
        elif "K_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_K_K)
        elif "Mg_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Mg_K)
        elif "Mn_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Mn_K)
        elif "Mn_La" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Mn_L)
        elif "Mg_La" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Mn_L)
        elif "N_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_N_K)
        elif "Na_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Na_K)
        elif "Ni_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Ni_K)
        elif "Ni_La" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Ni_L)     
        elif "O_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_O_K)       
        elif "P_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_P_K)
        elif "Pt_La" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Pt_L)
        elif "Pt_Ma" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Pt_M)
        elif "S_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_S_K)
        elif "Si_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Si_K)
        elif "Sn_La" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Si_K)
        elif "Ti_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Ti_K)    
        elif "Zr_Ka" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Zr_K)
        elif "Zr_La" in results[line].metadata.Sample.xray_lines:
            kfactors.append(kf_Zr_L)
        
    return kfactors
