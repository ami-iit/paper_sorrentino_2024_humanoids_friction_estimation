classdef UKS < handle
    % Uscented Kalman Smoother
    properties
        P,
        state_size;
        measurement_size;
        R, Q;
        f_trans;
        f_meas;
        l, w_m, w_c; % sigma_points_weights;
    end
    
    methods
        function obj = UKS(input)
            % initial_covariance : initial uncertainty
            % measurement_noise  : measurement noise covariance
            % process_noise      : process noise covariance
            % f_trans            : transition function (as function handle @)
            % f_meas             : measurement function (as function handle @)
            arguments
                input.initial_covariance;
                input.measurement_noise;
                input.process_noise;
                input.f_trans;
                input.f_meas;
            end
            obj.state_size = size(input.process_noise, 1);
            obj.P = input.initial_covariance;
            obj.measurement_size = size(input.measurement_noise, 1);
            obj.Q = input.process_noise;
            obj.R = input.measurement_noise;
            obj.f_trans = input.f_trans;
            obj.f_meas = input.f_meas;
            obj.get_weights();
        end
        
        function [xs, Ps] = smoothing_step(obj, x0, zs, us)
            % x0 : initial state
            % zs : sequence of measurements
            % us : sequence of inputs
            % xs : sequence of filtered states
            % Ps : sequence of covariances
            [xs, Ps] = obj.batch_filter(x0, zs, us);
            [xs, Ps] = obj.back_step(xs, Ps, us);
        end
        
        
        function [xs, Ps] = batch_filter(obj, x0, zs, us)
            % x0 : initial state
            % zs : sequence of measurements
            % us : sequence of inputs
            % xs : sequence of filtered states
            % Ps : sequence of covariances
            xs = cell(numel(us),1);
            xs{1} = x0;
            Ps = cell(numel(us),1);
            Ps{1} = obj.P;
            % making the data structs consistent
            zs = reshape(zs, [min(size(zs)), max(size(zs))]);
            us = reshape(us, [min(size(us)), max(size(us))]);
            for i = 2 : max(size(zs))
                [xs{i}, Ps{i}] = obj.step(xs{i-1}, us(:, i -1), zs(:,i));
            end
        end
        
        function [xs, Ps] = back_step(obj, xs, Ps, us)
            % xs : sequence of filtered states
            % Ps : sequence of covariances
            % us : sequence of inputs
            % making the data structs consistent
            us = reshape(us, [min(size(us)), max(size(us))]);
            for k = numel(xs)-1 : -1 : 1
                X = obj.get_sigma_p(xs{k}, Ps{k});
                % propagate sigma points
                X_p = zeros(obj.state_size, obj.state_size*2 + 1);
                for i = 1 : obj.state_size*2 + 1
                    X_p(:, i) = obj.f_trans(X(:, i), us(:, k));
                end
                [xb, Pb] = obj.get_mean_and_cov(X_p);
                Pb = Pb + obj.Q;
                P_xy = (X - xs{k}) * diag(obj.w_c) * (X_p - xb)';
                K = P_xy / Pb;
                xs{k} = xs{k} + K * (xs{k+1} - xb);
                Ps{k} = Ps{k} + K * (Ps{k+1} - Pb) * K';
            end
        end
        
        function [x, P] = step(obj, x, u, z)
            % x = state
            % u = input
            % z = measurement
            
            % --- Prediction step ---
            % get sigma points
            % obj.test_mean_and_cov(x, obj.P); % Decomment to
            % debug sigma points generation
            X = obj.get_sigma_p(x, obj.P);
            % propagate sigma points
            X_p = zeros(obj.state_size, obj.state_size*2 + 1);
            for i = 1 : obj.state_size*2 + 1
                X_p(:, i) = obj.f_trans(X(:, i), u);
            end
            % obtain a priori estimation
            [x_k_, P_k_] = obj.get_mean_and_cov(X_p);
            P_k_ = P_k_ + obj.Q;
            % --- Update step ---
            % get sigma points from a priori estimation
            X = obj.get_sigma_p(x_k_, P_k_);
            Y = zeros(obj.measurement_size, obj.state_size*2 + 1);
            for i = 1 : obj.state_size*2  + 1
                Y(:, i) = obj.f_meas(X(:, i));
            end
            [y_k, P_y] = obj.get_mean_and_cov(Y);
            % covariance of the predicted measurement
            P_y = P_y + obj.R;
            % Cross covariance
            P_xy = (X_p - x_k_) *  diag(obj.w_c) * (Y - y_k)';
            % update of the state
            K = P_xy / P_y;
            x = x_k_ + K * (z - y_k);
            obj.P = P_k_ - K * P_y * K';
            P = obj.P;
        end
        
        function X = get_sigma_p(obj, x, P)
            st_dim = numel(x);
            X = [x];
            sqrtP = chol((st_dim + obj.l) * P,'upper');
            for i = 1 : st_dim
                X = [X, x + sqrtP(i,:)'];
            end
            for j = 1 : st_dim
                X = [X, x - sqrtP(j,:)'];
            end
        
        end
        
        function [x, P] = get_mean_and_cov(obj, X)
            x = X * obj.w_m;
            P = (X - x) * diag(obj.w_c) * (X - x)';
        end
        
        function test_mean_and_cov(obj, x, P)
            % just for testing
            st_dim = numel(x);
            X = obj.get_sigma_p(x, P);
            [x_, P_] = obj.get_mean_and_cov(X);
            if any(abs(x - x_) > 1e-6)
                disp('mean not equivalent');
            end
            if any(any(abs(P - P_) > 1e-6))
                disp('covariance not equivalent');
            end
        end
        
        function get_weights(obj)
            % hardcoding the parameteers for computing the sigma points
            % since they don't need to be tuned
            k = 0;
            a = 1e-3;
            b = 2;
            obj.l = a^2 * obj.state_size;
            obj.w_m = ones(obj.state_size * 2 + 1, 1) * 1 / (2 * (obj.state_size + obj.l));
            obj.w_c = obj.w_m;
            obj.w_m(1) = obj.l / (obj.state_size + obj.l);
            obj.w_c(1) = obj.w_m(1) + (1 - a^2 + b);
        end
    end
end
