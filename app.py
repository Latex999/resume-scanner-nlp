"""
Resume Scanner with NLP - Main Application

This is the main entry point for the resume scanner application.
It provides both a command-line interface and launches the web API.
"""

import os
import sys
import argparse
from pathlib import Path
import json
import uvicorn
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/resume_scanner.log", rotation="10 MB", level="DEBUG", retention="1 week")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Resume Scanner with NLP")
    
    # Define CLI arguments
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Parse resume command
    parse_resume_parser = subparsers.add_parser("parse-resume", help="Parse a resume file")
    parse_resume_parser.add_argument("resume_file", help="Path to the resume file")
    parse_resume_parser.add_argument("--output", "-o", help="Output file for parsed resume (JSON)")
    
    # Parse job description command
    parse_job_parser = subparsers.add_parser("parse-job", help="Parse a job description")
    parse_job_parser.add_argument("job_file", help="Path to the job description file")
    parse_job_parser.add_argument("--output", "-o", help="Output file for parsed job (JSON)")
    
    # Match resume to job command
    match_parser = subparsers.add_parser("match", help="Match a resume to a job description")
    match_parser.add_argument("resume_file", help="Path to the resume file")
    match_parser.add_argument("job_file", help="Path to the job description file")
    match_parser.add_argument("--output", "-o", help="Output file for match results (JSON)")
    
    # Enhance resume command
    enhance_parser = subparsers.add_parser("enhance", help="Get enhancement suggestions for a resume")
    enhance_parser.add_argument("resume_file", help="Path to the resume file")
    enhance_parser.add_argument("job_file", help="Path to the job description file")
    
    # Serve API command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to run the server on")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "parse-resume":
        parse_resume(args)
    elif args.command == "parse-job":
        parse_job(args)
    elif args.command == "match":
        match_resume_to_job(args)
    elif args.command == "enhance":
        enhance_resume(args)
    elif args.command == "serve":
        serve_api(args)
    else:
        # Default to serving the API if no command is specified
        parser.print_help()

def parse_resume(args):
    """Parse a resume file."""
    from src.resume_parser.parser import ResumeParser
    
    logger.info(f"Parsing resume file: {args.resume_file}")
    
    try:
        # Initialize parser
        parser = ResumeParser()
        
        # Parse resume
        parsed_data = parser.parse_resume(args.resume_file)
        
        # Output results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(parsed_data, f, indent=2)
            logger.info(f"Parsed resume saved to {args.output}")
        else:
            print(json.dumps(parsed_data, indent=2))
    except Exception as e:
        logger.error(f"Error parsing resume: {e}")
        sys.exit(1)

def parse_job(args):
    """Parse a job description file."""
    from src.job_parser.parser import JobDescriptionParser
    
    logger.info(f"Parsing job description file: {args.job_file}")
    
    try:
        # Initialize parser
        parser = JobDescriptionParser()
        
        # Read job description from file
        with open(args.job_file, "r", encoding="utf-8") as f:
            job_text = f.read()
        
        # Parse job description
        parsed_data = parser.parse_job_description(job_text)
        
        # Output results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(parsed_data, f, indent=2)
            logger.info(f"Parsed job description saved to {args.output}")
        else:
            print(json.dumps(parsed_data, indent=2))
    except Exception as e:
        logger.error(f"Error parsing job description: {e}")
        sys.exit(1)

def match_resume_to_job(args):
    """Match a resume to a job description."""
    from src.resume_parser.parser import ResumeParser
    from src.job_parser.parser import JobDescriptionParser
    from src.matcher.matcher import ResumeMatcher
    
    logger.info(f"Matching resume {args.resume_file} to job {args.job_file}")
    
    try:
        # Initialize parsers and matcher
        resume_parser = ResumeParser()
        job_parser = JobDescriptionParser()
        matcher = ResumeMatcher()
        
        # Parse resume
        parsed_resume = resume_parser.parse_resume(args.resume_file)
        
        # Read and parse job description
        with open(args.job_file, "r", encoding="utf-8") as f:
            job_text = f.read()
        parsed_job = job_parser.parse_job_description(job_text)
        
        # Match resume to job
        match_results = matcher.calculate_match_score(parsed_resume, parsed_job)
        
        # Generate enhancement suggestions
        suggestions = matcher.generate_improvement_suggestions(
            parsed_resume, parsed_job, match_results
        )
        
        # Add suggestions to results
        match_results["improvement_suggestions"] = suggestions
        
        # Output results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(match_results, f, indent=2)
            logger.info(f"Match results saved to {args.output}")
        else:
            print(json.dumps(match_results, indent=2))
            
            # Print a more user-friendly summary
            print("\n" + "=" * 50)
            print(f"Match Score: {match_results['overall_score'] * 100:.1f}%")
            print("=" * 50)
            print(f"Summary: {match_results['summary']}")
            print(f"Recommendation: {match_results['recommendation']}")
            
            if suggestions:
                print("\nImprovement Suggestions:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"{i}. {suggestion}")
    except Exception as e:
        logger.error(f"Error matching resume to job: {e}")
        sys.exit(1)

def enhance_resume(args):
    """Get enhancement suggestions for a resume."""
    from src.resume_parser.parser import ResumeParser
    from src.job_parser.parser import JobDescriptionParser
    from src.matcher.matcher import ResumeMatcher
    
    logger.info(f"Generating enhancement suggestions for resume {args.resume_file}")
    
    try:
        # Initialize parsers and matcher
        resume_parser = ResumeParser()
        job_parser = JobDescriptionParser()
        matcher = ResumeMatcher()
        
        # Parse resume
        parsed_resume = resume_parser.parse_resume(args.resume_file)
        
        # Read and parse job description
        with open(args.job_file, "r", encoding="utf-8") as f:
            job_text = f.read()
        parsed_job = job_parser.parse_job_description(job_text)
        
        # Match resume to job
        match_results = matcher.calculate_match_score(parsed_resume, parsed_job)
        
        # Generate enhancement suggestions
        suggestions = matcher.generate_improvement_suggestions(
            parsed_resume, parsed_job, match_results
        )
        
        # Get ATS compatibility score
        ats_results = matcher.get_ats_compatibility_score(parsed_resume, parsed_job)
        
        # Print results
        print("\n" + "=" * 50)
        print("Resume Enhancement Suggestions")
        print("=" * 50)
        
        print(f"\nCurrent Match Score: {match_results['overall_score'] * 100:.1f}%")
        print(f"ATS Compatibility Score: {ats_results['score'] * 100:.1f}%")
        
        print("\nContent Improvement Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
        
        print("\nATS Optimization Suggestions:")
        for i, suggestion in enumerate(ats_results['suggestions'], 1):
            print(f"{i}. {suggestion}")
    except Exception as e:
        logger.error(f"Error generating enhancement suggestions: {e}")
        sys.exit(1)

def serve_api(args):
    """Start the API server."""
    logger.info(f"Starting API server on {args.host}:{args.port}")
    
    try:
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        # Start the API server
        uvicorn.run(
            "src.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Run the main function
    main()