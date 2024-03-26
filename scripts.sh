# 本地预览，网址为 localhost:1313
hugo server -D
# 生成public文件夹
hugo
# 每次生成的public都是全新的，会覆盖原来的。
hugo -F --cleanDestinationDir 