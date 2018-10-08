# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 19:02:32 2017

@author: Corentin
"""
def correct_result_2D(result):
    
    import copy
    result_cor = copy.deepcopy(result)
    
    for i in range (0, len(result_cor)):
        if "Ca_La" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*0
        elif "Cr_La" in result_cor[i].metadata.Sample.xray_lines:
            result_cor[i].data = result[i].data*0
        elif "Cu_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*0
        elif "Cu_La" in result_cor[i].metadata.Sample.xray_lines:
            result_cor[i].data = result[i].data*0

        #elif "Fe_La" in result_cor[i].metadata.Sample.xray_lines:
        #    result_cor[i].data = result[i].data*0

        elif "Fe_Lb3" in result_cor[i].metadata.Sample.xray_lines:
            result_cor[i].data = result[i].data*0
        elif "Fe_Ll" in result_cor[i].metadata.Sample.xray_lines:
            result_cor[i].data = result[i].data*0
        elif "Fe_Ln" in result_cor[i].metadata.Sample.xray_lines:
            result_cor[i].data = result[i].data*0

        elif "Ga_Ka" in result_cor[i].metadata.Sample.xray_lines:
            result_cor[i].data = result[i].data*0
        elif "Ga_La" in result_cor[i].metadata.Sample.xray_lines:
            result_cor[i].data = result[i].data*0
        elif "Mn_La" in result_cor[i].metadata.Sample.xray_lines:
            result_cor[i].data = result[i].data*0
        elif "Ni_La" in result_cor[i].metadata.Sample.xray_lines:
            result_cor[i].data = result[i].data*0
        elif "Pt_La" in result_cor[i].metadata.Sample.xray_lines:
            result_cor[i].data = result[i].data*0            
        elif "Pt_Ma" in result_cor[i].metadata.Sample.xray_lines:
            result_cor[i].data = result[i].data*0
                
    for i in range (0, len(result_cor)):
        if "Al_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*1.0131
        elif "Ca_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*1.112
        elif "Ca_La" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*2.23
        elif "Cr_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*1.134
        #elif "Cr_La" in result_cor[i].metadata.Sample.xray_lines: 
        #    result_cor[i].data = result[i].data*1.9565
        elif "Fe_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*1.127
        #elif "Fe_La" in result_cor[i].metadata.Sample.xray_lines: 
        #    result_cor[i].data = result[i].data*1.458
        elif "F_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data
        elif "K_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*1.1039    
        elif "Mg_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*1.01    
        elif "Mn_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*1.1252
        #elif "Mn_La" in result_cor[i].metadata.Sample.xray_lines: 
        #    result_cor[i].data = result[i].data*1.6059
        elif "Na_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*1.01            
        elif "Ni_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*1.1277            
        #elif "Ni_La" in result_cor[i].metadata.Sample.xray_lines: 
        #    result_cor[i].data = result[i].data*1.40795        
        elif "P_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*1.0498
        elif "S_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*1.06525
        elif "Si_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*1.0277
        elif "Zr_Ka" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*1.15
        elif "Zr_La" in result_cor[i].metadata.Sample.xray_lines: 
            result_cor[i].data = result[i].data*1.5208
    return result_cor
