function [NFC1, NFC2, NFC3] = NFC_train(generate_flag, validate_flag, n0, w_range, q_range)
    
    if nargin == 0
        load sdreDataset.mat dataset
    elseif generate_flag
        dataset = get_dataset();
        save sdreDataset.mat dataset
    else
        dataset = get_dataset(n0, w_range, q_range);
        save sdreDataset.mat dataset
    end
    if nargin <= 1
        validate_flag = 1;
    end
        
    init_fis = readfis('ts_control.fis');
    opt1 = anfisOptions('InitialFIS', init_fis, 'EpochNumber', 315, ...
        'InitialStepSize', 0.1, 'DisplayANFISInformation', 0, ...
        'DisplayFinalResults', 0, 'DisplayErrorValues', 0, ...
        'DisplayStepSize', 0);
    opt2 = opt1;
    %opt2.EpochNumber = 67;
    opt3 = opt1;
    %opt3.EpochNumber = 115;
    if validate_flag
        load sdreVal.mat sdreVal
        opt1.ValidationData = squeeze(sdreVal(:, 1, :));
        opt2.ValidationData = squeeze(sdreVal(:, 2, :));
        opt3.ValidationData = squeeze(sdreVal(:, 3, :));
    end
    
    tic
    NFC1 = anfis(squeeze(dataset(:, 1, :)), opt1);
    NFC2 = anfis(squeeze(dataset(:, 2, :)), opt2);
    NFC3 = anfis(squeeze(dataset(:, 3, :)), opt3);
    toc
end