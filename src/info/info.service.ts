import { Injectable, Logger } from '@nestjs/common';
import { simpleGit } from 'simple-git';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';

const execAsync = promisify(exec);

@Injectable()
export class InfoService {
  private readonly logger = new Logger(InfoService.name);

  async getProjectInfo(repoUrl: string, question: string, topN: number = 5): Promise<Object[]> {
    this.logger.log(`Starting getProjectInfo with repoUrl=${repoUrl}, question=${question}, topN=${topN}`);
    const localDirectory = './local-directory';

    // Очистка папки перед клонированием
    await this.clearDirectory(localDirectory);

    this.logger.log(`Cloning repository from ${repoUrl} to ${localDirectory}`);
    await simpleGit().clone(repoUrl, localDirectory);

    this.logger.log(`Deleting .git from ${localDirectory}`);
    await this.removeGitDirectory(localDirectory);

    this.logger.log(`Running Python script for combiner files in ${localDirectory}`);
    await execAsync('python ./file_combiner.py');

    this.logger.log(`Running Python script for question=${question}`);
    const result = await this.runGenerateTextScript('./output_file.txt', question);

    this.logger.log(`Python script completed with result: ${result}`);
    return [{ info: result }];
  }

  async clearDirectory(directoryPath: string): Promise<void> {
    this.logger.log(`Clearing directory ${directoryPath}`);
    if (fs.existsSync(directoryPath)) {
      const files = fs.readdirSync(directoryPath);
      for (const file of files) {
        const filePath = path.join(directoryPath, file);
        if (fs.lstatSync(filePath).isDirectory()) {
          await this.clearDirectory(filePath);
          fs.rmdirSync(filePath);
        } else {
          fs.unlinkSync(filePath);
        }
      }
    } else {
      fs.mkdirSync(directoryPath, { recursive: true });
    }
    this.logger.log(`Directory ${directoryPath} cleared`);
  }

  async removeGitDirectory(directoryPath: string): Promise<void> {
    const gitPath = path.join(directoryPath, '.git');
    if (fs.existsSync(gitPath)) {
      this.logger.log(`Removing .git directory at ${gitPath}`);
      await this.clearDirectory(gitPath); // Очистка содержимого папки .git
      fs.rmdirSync(gitPath); // Удаление папки .git
      this.logger.log(`.git directory removed`);
    } else {
      this.logger.log(`.git directory not found at ${gitPath}`);
    }
  }

  async runGenerateTextScript(directoryPath: string, question: string, topN = 10): Promise<string> {
    const scriptPath = './generate_text.py';

    try {
      const { stdout, stderr } = await execAsync(`python ${scriptPath} ${directoryPath} "${question}" --top_n ${topN}`);
      if (stderr) {
        this.logger.error(`Python script error: ${stderr}`);
        return `Error: ${stderr}`;
      }
      this.logger.log(`Python script output: ${stdout}`);
      return stdout;
    } catch (error) {
      this.logger.error(`Execution error: ${error}`);
      return `Execution error: ${error}`;
    }
  }
}