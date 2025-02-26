text_tsir = '''
Armadillo RLIT[1] @ PSNR: 30.173
Armadillo RLIT[1] @ SSIM: 0.9610
Armadillo RLIT[1] @ LPIPS: 0.0492
Armadillo RLIT[2] @ PSNR: 29.513
Armadillo RLIT[2] @ SSIM: 0.9582
Armadillo RLIT[2] @ LPIPS: 0.0563
Armadillo RLIT[3] @ PSNR: 30.304
Armadillo RLIT[3] @ SSIM: 0.9625
Armadillo RLIT[3] @ LPIPS: 0.0454
Armadillo RLIT[4] @ PSNR: 30.324
Armadillo RLIT[4] @ SSIM: 0.9638
Armadillo RLIT[4] @ LPIPS: 0.0526
Armadillo RLIT[5] @ PSNR: 31.256
Armadillo RLIT[5] @ SSIM: 0.9683
Armadillo RLIT[5] @ LPIPS: 0.0432
Armadillo Albedo @ PSNR: 30.398
Armadillo Albedo @ SSIM: 0.9332
Armadillo Albedo @ LPIPS: 0.1004
Ficus RLIT[1] @ PSNR: 21.718
Ficus RLIT[1] @ SSIM: 0.9246
Ficus RLIT[1] @ LPIPS: 0.0695
Ficus RLIT[2] @ PSNR: 22.418
Ficus RLIT[2] @ SSIM: 0.9225
Ficus RLIT[2] @ LPIPS: 0.0721
Ficus RLIT[3] @ PSNR: 20.863
Ficus RLIT[3] @ SSIM: 0.9239
Ficus RLIT[3] @ LPIPS: 0.0733
Ficus RLIT[4] @ PSNR: 21.710
Ficus RLIT[4] @ SSIM: 0.9244
Ficus RLIT[4] @ LPIPS: 0.0699
Ficus RLIT[5] @ PSNR: 20.884
Ficus RLIT[5] @ SSIM: 0.9262
Ficus RLIT[5] @ LPIPS: 0.0707
Ficus Albedo @ PSNR: 25.614
Ficus Albedo @ SSIM: 0.9148
Ficus Albedo @ LPIPS: 0.0873
Hotdog RLIT[1] @ PSNR: 25.766
Hotdog RLIT[1] @ SSIM: 0.9288
Hotdog RLIT[1] @ LPIPS: 0.0988
Hotdog RLIT[2] @ PSNR: 24.534
Hotdog RLIT[2] @ SSIM: 0.9350
Hotdog RLIT[2] @ LPIPS: 0.0904
Hotdog RLIT[3] @ PSNR: 29.112
Hotdog RLIT[3] @ SSIM: 0.9479
Hotdog RLIT[3] @ LPIPS: 0.0689
Hotdog RLIT[4] @ PSNR: 28.387
Hotdog RLIT[4] @ SSIM: 0.9407
Hotdog RLIT[4] @ LPIPS: 0.0837
Hotdog RLIT[5] @ PSNR: 28.782
Hotdog RLIT[5] @ SSIM: 0.9531
Hotdog RLIT[5] @ LPIPS: 0.0656
Hotdog Albedo @ PSNR: 26.043
Hotdog Albedo @ SSIM: 0.9409
Hotdog Albedo @ LPIPS: 0.0828
Lego RLIT[1] @ PSNR: 25.437
Lego RLIT[1] @ SSIM: 0.8888
Lego RLIT[1] @ LPIPS: 0.0921
Lego RLIT[2] @ PSNR: 24.856
Lego RLIT[2] @ SSIM: 0.8925
Lego RLIT[2] @ LPIPS: 0.0874
Lego RLIT[3] @ PSNR: 26.826
Lego RLIT[3] @ SSIM: 0.9179
Lego RLIT[3] @ LPIPS: 0.0763
Lego RLIT[4] @ PSNR: 26.266
Lego RLIT[4] @ SSIM: 0.9082
Lego RLIT[4] @ LPIPS: 0.0774
Lego RLIT[5] @ PSNR: 27.125
Lego RLIT[5] @ SSIM: 0.9183
Lego RLIT[5] @ LPIPS: 0.0735
Lego Albedo @ PSNR: 21.878
Lego Albedo @ SSIM: 0.8418
Lego Albedo @ LPIPS: 0.1299
'''

if __name__ == '__main__':

    def avg(l, fixed):
        fmt = f'.{fixed}f'
        return f'{sum(l) / len(l):{fmt}}'

    def display(l):
        print(' & '.join([str(x) for x in l]))

    lut = {}
    for key, value in [line.split(': ') for line in text_tsir.split('\n') if line]:
        lut.setdefault(key.split(' ', 1)[0], {})[key.split(' ', 1)[1]] = float(value)

    scenes = ['Lego', 'Hotdog', 'Armadillo', 'Ficus']

    print('Relight')
    lst = []
    for scene in scenes:
        lst.append(round(sum([lut[scene][f'RLIT[{i+1}] @ PSNR'] for i in range(5)]) / 5, 3))
        lst.append(round(sum([lut[scene][f'RLIT[{i+1}] @ SSIM'] for i in range(5)]) / 5, 4))
        lst.append(round(sum([lut[scene][f'RLIT[{i+1}] @ LPIPS'] for i in range(5)]) / 5, 4))
    lst.append(avg(lst[0::3], 2))
    lst.append(avg(lst[1::3], 4))
    lst.append(avg(lst[2::3], 4))
    display(lst)

    print('Albedo')
    lst = []
    for scene in scenes:
        lst.append(lut[scene]['Albedo @ PSNR'])
        lst.append(lut[scene]['Albedo @ SSIM'])
        lst.append(lut[scene]['Albedo @ LPIPS'])
    lst.append(avg(lst[0::3], 2))
    lst.append(avg(lst[1::3], 4))
    lst.append(avg(lst[2::3], 4))
    display(lst)