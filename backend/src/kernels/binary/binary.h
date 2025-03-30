#define OP_MUL           0
#define OP_ADD           1
#define OP_SUB           2
#define OP_DIV           3
#define OP_DIV_NO_NAN    4
#define OP_DIV_REAL      5
#define OP_MAXIMUM       6
#define OP_MINIMUM       7
#define OP_POW           8
#define OP_SQR_DIFF      9
#define OP_TANH_GRAD     10
#define OP_RELU_GRAD     11

TYPE_0 power(TYPE_0 x, TYPE_0 y){
    if(y == TYPE_0(0)){
        return TYPE_0(1);
    }else if (x == TYPE_0(0)){
        return TYPE_0(0);
    }

    #if TYPE_NUM_0 == FLOAT_NUM
        return pow(x, y);
    #else
        TYPE_P_0 res = x;
        for(int i = 0; i < int(y) - 1; i++){
            res *= x;
        }

        return res;
    #endif
}

TYPE_0 apply_op(in TYPE_0 X, in TYPE_0 Y, uint op){
    switch(op){
        case OP_MUL:
            return X * Y;
        case OP_ADD:
            return X + Y;
        case OP_SUB:
            return X - Y;
        case OP_DIV:
            return X / Y;
        case OP_DIV_NO_NAN:
            if(Y == TYPE_0(0)){
                return TYPE_0(0);
            }else{
                return X / Y;
            }
        case OP_DIV_REAL:
            return X / Y;
        case OP_MAXIMUM:
            return max(X, Y);
        case OP_MINIMUM:
            return min(X, Y);
        case OP_POW:
            return power(X, Y);
        case OP_SQR_DIFF:
            return (X - Y) * (X - Y);
        case OP_TANH_GRAD:
            return Y * (TYPE_0(1) - X * X);
        case OP_RELU_GRAD:
            if(Y > TYPE_0(0)){
                return X;
            }else{
                return TYPE_0(0);
            }
    }
}