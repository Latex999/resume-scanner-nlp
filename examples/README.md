# Example Files for Resume Scanner

This directory contains example files that you can use to test the resume scanner functionality:

## Files Included

- `example_resume.txt`: A sample resume in plain text format
- `example_job_description.txt`: A sample job description in plain text format

## How to Use

You can use these files to test the resume scanner with the following commands:

### Parse a resume

```bash
python app.py parse-resume examples/example_resume.txt --output parsed_resume.json
```

### Parse a job description

```bash
python app.py parse-job examples/example_job_description.txt --output parsed_job.json
```

### Match a resume to a job description

```bash
python app.py match examples/example_resume.txt examples/example_job_description.txt
```

### Get enhancement suggestions

```bash
python app.py enhance examples/example_resume.txt examples/example_job_description.txt
```

## Next Steps

After testing with these examples, you can try using your own resume and job descriptions to see how they match.

Remember that the quality of the match depends on how well the resume and job description are structured and formatted.