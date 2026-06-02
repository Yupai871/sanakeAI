# 学生运行结果提交说明

每位同学只提交自己的结果文件，文件名使用：

```text
学号_姓名.csv
```

示例：

```text
20260001_张三.csv
```

不要修改其他同学的文件。

## 结果字段

请参考 `template.csv`，至少填写 5 次运行结果：

```text
student_id,name,run_id,model_file,score,steps,notes
```

字段说明：

- `student_id`：学号
- `name`：姓名
- `run_id`：第几次运行，从 1 开始
- `model_file`：使用的模型，例如 `model_v2/best_model.pth`
- `score`：本局得分
- `steps`：本局运行步数；如果没有统计可以填 `unknown`
- `notes`：观察到的现象，例如“撞墙结束”“吃到 20 个食物”

## 提交要求

1. 每位同学新建自己的分支。
2. 只新增或修改自己的 `student_results/学号_姓名.csv`。
3. 提交信息格式：

```text
submit: 学号 姓名 snake ai result
```

示例：

```text
submit: 20260001 张三 snake ai result
```
