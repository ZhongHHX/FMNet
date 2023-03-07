
def val():
    global one
    print("val({})".format(one))
    one = one + 1
    print("val({})".format(one))


if __name__ == '__main__':
    one = 0
    # print(testnum)
    for i in range(0, 5):
        val()

    print(one)

    val()