# 在 tiling 实现文件中查找 TilingParse<TYPE>
# 注意：文件可能在 arch32/ 或 arch35/ 目录下，根据芯片架构确定
find op_host -name "*_tiling*.cpp" -exec grep -l "TilingParse" {} \;
