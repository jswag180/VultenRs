#define OP_MUL           0
#define OP_ADD           1
#define OP_SUB           2
#define OP_DIV           3
#define OP_DIV_NO_NAN    4
#define OP_DIV_REAL      5
#define OP_MAXIMUM       6
#define OP_MINIMUM       7
#define OP_POW           8

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