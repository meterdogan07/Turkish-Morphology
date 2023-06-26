
with open("/kuacc/users/merdogan18/hpc_run/TrMor2023/versions/prefix-tuning/xout_validation5_run.txt",'r',encoding = "utf-8") as f:
    a = f.readlines()
    true_val = 0
    false_val = 0
    for line in a:
        ix = line.find("=>")
        if(line[0:ix-1].strip() == line[ix+3:].strip()):
            true_val += 1
        else:
            false_val += 1
            print(line[0:ix-1].strip() + "   " +  line[ix+3:].strip())
print(100*true_val/(true_val+false_val))


