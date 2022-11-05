# 该文件用于测量多个视频文件的大小。

import os

TOTAL_VIDEO_CHUNK = 49                                   # 总视频块设置为49个
BITRATE_LEVELS = 6                                       # 码率水平一共设置为6个，详情见main.py
VIDEO_PATH = './video_server/'                           # 设置视频路径
VIDEO_FOLDER = 'video'

# 这里设置：所有视频都在'video_server/video[1, 2, 3, 4, 5]'目录下
# 且video5目录下的视频质量最差，video1目录下的视频质量最好


def get_video_size():
    for bitrate in range(BITRATE_LEVELS):
        with open('video_size_' + str(bitrate), 'w') as file:           # 对每一个码率水平，都新创建一个文件进行记录，这个文件保存在当前目录下
            for chunk_num in range(1, TOTAL_VIDEO_CHUNK + 1):
                video_chunk_path = VIDEO_PATH + VIDEO_FOLDER \
                                   + str(BITRATE_LEVELS - bitrate) \
                                   + '/' \
                                   + str(chunk_num) + '.m4s'
                chunk_size = os.path.getsize(video_chunk_path)
                file.write(str(chunk_size) + '\n')                      # 每一行都记录一个视频文件的大小

# 最终会得到'video_size_0'到'video_size_5'一共六个记录文件大小的文件
# 在'video_size_1'文件中，记录的是video5目录下视频文件的大小

# 16行中，本来应该是以'wb'二进制写的，但是这会引起警告，因此我改为了'w'形式打开文件
