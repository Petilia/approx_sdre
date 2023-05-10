function clear_data()
    load('sdreDataset.mat')
    remove_list = [];
    for k = 1:5050
       if all(dataset(k, :, 1:2) < 0.0002, 'all') 
            remove_list = [remove_list, k];
       end
    end
    dataset(remove_list, :, :) = [];
    save clearData.mat dataset
end