function compare_with_approx_control(train_flag, T, plot_flag, velocity0, angle0)
    % функция для сравнения SDRE и его нечеткой аппроксимации
    if nargin == 0
        train_flag = 0;
        T = 100;
        plot_flag = 1;
        velocity0 = [0.1301, 0.1637, 0.1174]';
        angle0 = [0.1881, 0.1131, 0.0414, 0.9747]';
    elseif nargin == 1
        T = 100;
        plot_flag = 1;
        velocity0 = [0.1301, 0.1637, 0.1174]';
        angle0 = [0.1881, 0.1131, 0.0414, 0.9747]';
    elseif nargin == 2
        plot_flag = 1;
        velocity0 = [0.1301, 0.1637, 0.1174]';
        angle0 = [0.1881, 0.1131, 0.0414, 0.9747]';
    elseif nargin <5
        velocity0 = [0.1301, 0.1637, 0.1174]';
        angle0 = [0.1881, 0.1131, 0.0414, 0.9747]';
    end
    
    if train_flag
        [u1, u2, u3] = NFC_train();
        writeFIS(u1, 'u1.fis');
        writeFIS(u2, 'u2.fis');
        writeFIS(u3, 'u3.fis');
    else
        u1 = readfis('u1.fis');
        u2 = readfis('u2.fis');
        u3 = readfis('u3.fis');
    end  
    nfc = @(x) [evalfis(u1, [x(1), x(4)]); evalfis(u2, [x(2), x(5)]); ...
        evalfis(u3, [x(3), x(6)])];
    
    tic     
    u_data(plot_flag, velocity0, angle0); % запуск SDRE
    toc

    tic 
    u_data(plot_flag, velocity0, angle0, nfc); % запуск nfc
    toc
end