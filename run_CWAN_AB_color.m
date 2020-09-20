function res = run_CWAN_AB_color(net, x, dzdy, res, varargin)

opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.accumulate = false ;
opts.cudnn = false ;
opts.backPropDepth = +inf ;

opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;

if (nargin <= 2) || isempty(dzdy)
    doder = false ;
else
    doder = true ;
end

if opts.cudnn
    cudnn = {'CuDNN'} ;
else
    cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
    res = struct(...
        'x', cell(1,n+1), ...
        'dzdx', cell(1,n+1), ...
        'dzdw', cell(1,n+1), ...
        'aux', cell(1,n+1), ...
        'time', num2cell(zeros(1,n+1)), ...
        'backwardTime', num2cell(zeros(1,n+1))) ;
end
res(1).x = x ;

for i=1:n
    l = net.layers{i} ;
    res(i).time = tic ;
    switch l.type
        case 'conv'
            if gpuMode == 1
                res(i+1).x = vl_nnconv(res(i).x, (l.weights{1}), (l.weights{2}), ...
                    'pad', l.pad, 'stride', l.stride, ...
                    cudnn{:}) ;
            else
                res(i+1).x = vl_nnconv(res(i).x, (l.weights{1}), (l.weights{2}), ...
                    'pad', l.pad, 'stride', l.stride, ...
                    cudnn{:}) ;
            end
            clear temp;
            if (i == 9)
                if gpuMode == 0
                    temp(:,:,3:4,:)=(res(i+1).x);
                    temp(:,:,1:2,:)=(res(1).x);
                    res(i+1).x=single(temp);
                else
                    temp(:,:,3:4,:)=(res(i+1).x);
                    temp(:,:,1:2,:)=(res(1).x);
                    res(i+1).x=single(temp);
                end
            end
            
            
        case 'convt'
            if isfield(l, 'weights')
                res(i+1).x = vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                    'crop', l.crop, 'upsample', l.upsample, ...
                    cudnn{:}) ;
            else
                res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, ...
                    'crop', l.pad, 'upsample', l.upsample, ...
                    cudnn{:}) ;
            end
        case 'pool'
            %         display('pool')
            %           tic
            res(i+1).x = vl_nnpool(res(i).x, l.pool, ...
                'pad', l.pad, 'stride', l.stride, ...
                'method', l.method, ...
                cudnn{:}) ;
            %                          toc
        case 'normalize'
            res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
        case 'softmax'
            res(i+1).x = vl_nnsoftmax(res(i).x) ;
        case 'loss'
            res(i+1).x = vl_nnloss(res(i).x, l.class) ;
        case 'softmaxloss'
            res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
        case 'Eucloss_Poly'
            res(i+1).x = euclideanloss_PolyFit(res(i).x, l.class) ;
        case 'Eucloss'
            res(i+1).x = euclideanloss(res(i).x, l.class) ;
        case 'Eucloss_2D'
            res(i+1).x = vl_nnL2(res(i).x, l.class) ;
        case 'Eucloss_Light'
            res(i+1).x = euclideanloss_Light(res(i).x, l.class) ;
        case 'L1'
            res(i+1).x = vl_nnloss_L1(res(i).x, l.class);
        case 'L1_AB_Loss'
            res(i+1).x = vl_nnloss_L1_AB(res(i).x, l.class);
        case 'L1_AB_Loss2'
            res(i+1).x = vl_nnloss_L1_AB2(res(i).x, l.class);
        case 'L1_loss'
            res(i+1).x = euclideanloss_Light_L1(res(i).x, l.class) ;
            %         res(i+1).x = vl_L1_Loss_Light(res(i).x, l.class) ;
        case 'vl_nnEcluSigmoid'
            res(i+1).x = vl_nnEcluSigmoid2(res(i).x, l.class) ;
        case 'relu'
            res(i+1).x = vl_nnrelu(res(i).x) ;
        case 'sigmoid'
            res(i+1).x = vl_nnsigmoid(res(i).x) ;
        case 'noffset'
            res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;
        case 'spnorm'
            res(i+1).x = vl_nnspnorm(res(i).x, l.param) ;
        case 'dropout'
            if opts.disableDropout
                res(i+1).x = res(i).x ;
            elseif opts.freezeDropout
                [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate, 'mask', res(i+1).aux) ;
            else
                [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
            end
        case 'bnorm'
            %         display('bnorm')
            %           tic
            if isfield(l, 'weights')
                res(i+1).x = vl_nnbnorm(res(i).x, (l.weights{1}), (l.weights{2})) ;
            else
                res(i+1).x = vl_nnbnorm(res(i).x, l.filters, l.biases) ;
            end
            %       toc
        case 'pdist'
            res(i+1) = vl_nnpdist(res(i).x, l.p, 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
        case 'custom'
            res(i+1) = l.forward(l, res(i), res(i+1)) ;
        otherwise
            error('Unknown layer type %s', l.type) ;
    end
    % optionally forget intermediate results
    forget = opts.conserveMemory ;
    forget = forget & (~doder || strcmp(l.type, 'relu')) ;
    forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
    forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
    if forget
        res(i).x = [] ;
    end
    if gpuMode & opts.sync
        % This should make things slower, but on MATLAB 2014a it is necessary
        % for any decent performance.
        wait(gpuDevice) ;
    end
    res(i).time = toc(res(i).time) ;
end

if doder
    res(n+1).dzdx = dzdy ;
    for i=n:-1:max(1, n-opts.backPropDepth+1)
        l = net.layers{i} ;
        res(i).backwardTime = tic ;
        switch l.type
            case 'conv'
                if ~opts.accumulate
                    if isfield(l, 'weights')
                        if (i == 9)
                            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                                res(i+1).dzdx(:,:,3:4,:), ...
                                'pad', l.pad, 'stride', l.stride, ...
                                cudnn{:}) ;
                        else
                            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                                res(i+1).dzdx, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                cudnn{:}) ;
                        end
                    end
                else
                    dzdw = cell(1,2) ;
                    if isfield(l, 'weights')
                        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                            vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                            res(i+1).dzdx, ...
                            'pad', l.pad, 'stride', l.stride, ...
                            cudnn{:}) ;
                    else
                        % Legacy code: will go
                        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                            vl_nnconv(res(i).x, l.filters, l.biases, ...
                            res(i+1).dzdx, ...
                            'pad', l.pad, 'stride', l.stride, ...
                            cudnn{:}) ;
                    end
                    for j=1:2
                        res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
                    end
                    clear dzdw ;
                end
                
            case 'convt'
                if ~opts.accumulate
                    if isfield(l, 'weights')
                        [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                            vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                            res(i+1).dzdx, ...
                            'crop', l.crop, 'upsample', l.upsample, ...
                            cudnn{:}) ;
                    else
                        % Legacy code: will go
                        [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                            vl_nnconvt(res(i).x, l.filters, l.biases, ...
                            res(i+1).dzdx, ...
                            'crop', l.crop, 'upsample', l.upsample, ...
                            cudnn{:}) ;
                    end
                else
                    dzdw = cell(1,2) ;
                    if isfield(l, 'weights')
                        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                            vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                            res(i+1).dzdx, ...
                            'crop', l.crop, 'upsample', l.upsample, ...
                            cudnn{:}) ;
                    else
                        % Legacy code: will go
                        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                            vl_nnconvt(res(i).x, l.filters, l.biases, ...
                            res(i+1).dzdx, ...
                            'crop', l.crop, 'upsample', l.upsample, ...
                            cudnn{:}) ;
                    end
                    for j=1:2
                        res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
                    end
                    clear dzdw ;
                end
                
            case 'pool'
                res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                    'pad', l.pad, 'stride', l.stride, ...
                    'method', l.method, ...
                    cudnn{:}) ;
            case 'normalize'
                res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
            case 'softmax'
                res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;
            case 'loss'
                res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
            case 'softmaxloss'
                res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
            case 'Eucloss_Poly'
                res(i).dzdx = euclideanloss_PolyFit(res(i).x, l.class, res(i+1).dzdx) ;
            case 'Eucloss'
                res(i).dzdx = euclideanloss(res(i).x, l.class, res(i+1).dzdx) ;
            case 'Eucloss_2D'
                res(i).dzdx = vl_nnL2(res(i).x, l.class, res(i+1).dzdx);
            case 'Eucloss_Light'
                res(i).dzdx = euclideanloss_Light(res(i).x, l.class, res(i+1).dzdx) ;
            case 'L1_loss'
                res(i).dzdx = euclideanloss_Light_L1(res(i).x, l.class, res(i+1).dzdx) ;
            case 'L1'
                res(i).dzdx = vl_nnloss_L1(res(i).x, l.class, res(i+1).dzdx);
            case 'L1_AB_Loss'
                res(i).dzdx = vl_nnloss_L1_AB(res(i).x, l.class, res(i+1).dzdx);
            case 'L1_AB_Loss2'
                res(i).dzdx = vl_nnloss_L1_AB2(res(i).x, l.class, res(i+1).dzdx);
            case 'vl_nnEcluSigmoid'
                res(i).dzdx = vl_nnEcluSigmoid2(res(i).x, l.class, res(i+1).dzdx) ;
            case 'relu'
                if ~isempty(res(i).x)
                    res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx) ;
                else
                    % if res(i).x is empty, it has been optimized away, so we use this
                    % hack (which works only for ReLU):
                    res(i).dzdx = vl_nnrelu(res(i+1).x, res(i+1).dzdx) ;
                end
            case 'sigmoid'
                res(i).dzdx = vl_nnsigmoid(res(i).x, res(i+1).dzdx) ;
            case 'noffset'
                res(i).dzdx = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;
            case 'spnorm'
                res(i).dzdx = vl_nnspnorm(res(i).x, l.param, res(i+1).dzdx) ;
            case 'dropout'
                if opts.disableDropout
                    res(i).dzdx = res(i+1).dzdx ;
                else
                    res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, ...
                        'mask', res(i+1).aux) ;
                end
            case 'bnorm'
                if ~opts.accumulate
                    if isfield(l, 'weights')
                        [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                            vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                            res(i+1).dzdx) ;
                    else
                        [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                            vl_nnbnorm(res(i).x, l.filters, l.biases, ...
                            res(i+1).dzdx) ;
                    end
                else
                    dzdw = cell(1,2) ;
                    if isfield(l, 'weights')
                        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                            vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                            res(i+1).dzdx) ;
                    else
                        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                            vl_nnbnorm(res(i).x, l.filters, l.biases, ...
                            res(i+1).dzdx) ;
                    end
                    for j=1:2
                        res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
                    end
                    clear dzdw ;
                end
            case 'pdist'
                res(i).dzdx = vl_nnpdist(res(i).x, l.p, res(i+1).dzdx, ...
                    'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
            case 'custom'
                res(i) = l.backward(l, res(i), res(i+1)) ;
        end
        if opts.conserveMemory
            res(i+1).dzdx = [] ;
        end
        if gpuMode & opts.sync
            wait(gpuDevice) ;
        end
        res(i).backwardTime = toc(res(i).backwardTime) ;
    end
end
