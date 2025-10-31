
#***************************************************************************************
entity_alignment_data = { 
    "dbp_yg": ["train", "data/dbp_yg/", "hit"],
    "dbp_wd": ["test", "data/dbp_wd/", "hit"]
}

# 3:1:1 directly use
new_deepmatcher_data = { 
    "m1":["train","data/em-wa/", "f1"],
    "m2":["test","data/em-ds/", "f1"],
    "m3":["train","data/em-fz/", "f1"],
    "m4":["train","data/em-ia/", "f1"],
}


entity_linking_data = {
    "t2d":["train","data/t2d/", "f1"],
    "limaya":["test", "data/Limaye/", "f1"]
}
