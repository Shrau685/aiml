def towerOfHanoi(n,src,dest,i):
    if n==0:
        return 
    towerOfHanoi(n-1,src,i,dest)
    print("move disk",n,"from rod",src,"to rod",dest)
    towerOfHanoi(n-1,i,dest,src)


n=3
towerOfHanoi(n,"A","B","C")