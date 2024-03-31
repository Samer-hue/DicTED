import numpy as np

def convert_emb_to_npy(input_emb, output_npy,size):
    # Read .emb file
    with open(input_emb, 'r') as emb_file:
        lines = emb_file.readlines()

    # Extracting rows
    data_lines = lines[1:]  # Skip the first line

    # Parse the data into a NumPy array
    data_array = np.array([list(map(float, line.split()[1:])) for line in data_lines])
    print(data_array.shape)
    print(data_array)

    # Add size zeros to the first row of the array
    zero_row = np.zeros((1, size))
    data_array = np.vstack([zero_row, data_array])

    # Save as .npy file
    np.save(output_npy, data_array)

# Call the function, passing in the.emb filename and the output.npy filename
def emb_to_npy(data_name, model_name,size=172):
   if model_name == 'deepwalk':
        convert_emb_to_npy("./processed/{}/ml_{}_node1.emb".format(data_name,data_name), "./processed/{}/ml_{}_node1.npy".format(data_name,data_name),size) 
   elif  model_name == 'gae':
        b = "./processed/{}/ml_{}_node_{}.npy".format(data_name,data_name,model_name)
        convert_emb_to_npy('./processed/{}/{}_GAE.emb'.format(data_name,data_name), b,size)
        
   elif model_name == 'MVGRL':
        b = "./processed/{}/ml_{}_node_{}.npy".format(data_name,data_name,model_name)
        convert_emb_to_npy('./processed/{}/{}_MVGRL.emb'.format(data_name,data_name), b,size)
   elif model_name == 'SDCN':
        b = "./processed/{}/ml_{}_node_{}.npy".format(data_name,data_name,model_name)
        convert_emb_to_npy('./processed/{}/{}_SDCN.emb'.format(data_name,data_name), b,size)
   elif model_name == 'TGN':
        b = "./processed/{}/ml_{}_node_{}.npy".format(data_name,data_name,model_name)
        convert_emb_to_npy('./processed/{}/{}_TGN.emb'.format(data_name,data_name), b,size)
   elif model_name == 'TREND':
        b = "./processed/{}/ml_{}_node_{}.npy".format(data_name,data_name,model_name)
        convert_emb_to_npy('./processed/{}/{}_TREND.emb'.format(data_name,data_name), b,size)
   else:
        a = "./processed/{}/ml_{}_{}.emb".format(data_name,data_name,model_name)
        b = "./processed/{}/ml_{}_node_{}.npy".format(data_name,data_name,model_name)
        convert_emb_to_npy(a, b,size)

if __name__ == "__main__":
   for data_name in ['uci','enron','socialevolve_1month']:
    for model_name in ['SDCN','TGN']:
        convert_emb_to_npy("./processed/{}/{}_{}.emb".format(data_name,data_name,model_name),"./processed/{}/ml_{}_node_{}.npy".format(data_name,data_name,model_name),172)