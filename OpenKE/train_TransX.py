import config
import models
import tensorflow as tf
import numpy as np

def train_transx(root='./task/',model='E'):
    con = config.Config()
    #Input training files from benchmarks/FB15K/ folder.
    con.set_in_path(root)
    #True: Input test files from the same folder.
    #con.set_test_link_prediction(True)
    #con.set_test_triple_classification(True)

    con.set_work_threads(8)
    con.set_train_times(500)
    con.set_nbatches(100)
    con.set_alpha(0.001)
    con.set_margin(1.0)
    con.set_bern(0)
    con.set_dimension(100)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("SGD")

    if model == 'R':
        con.init()
        con.set_model(models.TransE)
        con.run()
        parameters = con.get_parameters("numpy")
        conR = config.Config()
        #Input training files from benchmarks/FB15K/ folder.
        conR.set_in_path(root)
        #True: Input test files from the same folder.
        #conR.set_test_link_prediction(True)
        #conR.set_test_triple_classification(True)

        conR.set_work_threads(8)
        conR.set_train_times(500)
        conR.set_nbatches(100)
        conR.set_alpha(0.001)
        conR.set_bern(0)
        conR.set_dimension(100)
        conR.set_margin(1)
        conR.set_ent_neg_rate(1)
        conR.set_rel_neg_rate(0)
        conR.set_opt_method("SGD")

        #Models will be exported via tf.Saver() automatically.
        conR.set_export_files(root+'/'+model+'/'+"model.vec.tf", 0)
        #Model parameters will be exported to json files automatically.
        conR.set_out_files(root+'/'+model+'/'+model+".vec.json")
        #Initialize experimental settings.
        conR.init()
        #Load pretrained TransE results.
        conR.set_model(models.TransR)
        parameters["transfer_matrix"] = np.array([(np.identity(100).reshape((100*100))) for i in range(conR.get_rel_total())])
        conR.set_parameters(parameters)
        #Train the model.
        conR.run()
        #To test models after training needs "set_test_flag(True)".
        #conR.test()
        return None


    #Models will be exported via tf.Saver() automatically.
    con.set_export_files(root+'/'+model+'/'+"model.vec.tf", 0)
    #Model parameters will be exported to json files automatically.
    con.set_out_files(root+'/'+model+'/'+model+".vec.json")
    #Initialize experimental settings.
    con.init()
    #Set the knowledge embedding model
    con.set_model(models_dict[model])
    #Train the model.
    con.run()
    #To test models after training needs "set_test_flag(True)".
    #con.test()

models_dict = {
    'E':models.TransE,
    'D':models.TransD,
    'H':models.TransH,
    'R':models.TransR,
}
if __name__ == '__main__':
    for model in models_dict.keys():
        train_transx(model=model)
