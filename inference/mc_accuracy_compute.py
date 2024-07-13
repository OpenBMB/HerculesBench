

def min_edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    
    # 创建一个二维数组来保存编辑距离
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化第一列和第一行
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # 填充dp数组
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 不需要操作
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,     # 删除操作
                               dp[i][j - 1] + 1,     # 插入操作
                               dp[i - 1][j - 1] + 1) # 替换操作
    return dp[m][n]



def extract_first_letter(s, option_list, ground_truth):
    # 使用正则表达式查找符合 \\boxed{SUBSTRING} 格式的子字符串
    # substring = extract_boxed_content(s)
    # print(substring)
    if "\\boxed{" in s:
        substring = s.split("\\boxed{")[1]
        for char in substring:
            if char in option_list:
                return char
        
        min_dist_options = []
        for option, content in option_list.items():
            min_dist_options.append(min_edit_distance(substring, content))
        
        char = list(option_list.keys())[min_dist_options.index(min(min_dist_options))]
            

    elif "answer is" in s:
        substring = s.split("answer is")[1]
        for char in substring:
            if char in option_list:
                return char
        
        min_dist_options = []
        for option, content in option_list.items():
            min_dist_options.append(min_edit_distance(substring, content))
        
        char = list(option_list.keys())[min_dist_options.index(min(min_dist_options))]
            

    else:
        return None
        
    return char == ground_truth