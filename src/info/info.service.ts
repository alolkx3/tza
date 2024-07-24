import { Injectable, Logger } from '@nestjs/common';
import { simpleGit } from 'simple-git';
import * as fs from 'fs';
import * as path from 'path';
import {
  SupportedTextSplitterLanguages,
  RecursiveCharacterTextSplitter
} from 'langchain/text_splitter';
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { TextLoader } from "langchain/document_loaders/fs/text";

interface FileWithExtension {
  file: string;
  extension: string;
}

const embeddings = new OllamaEmbeddings({
  model: "llama2",
  baseUrl: "http://localhost:11434",
});

@Injectable()
export class InfoService {
  private readonly logger = new Logger(InfoService.name);

  async getProjectInfo(repoUrl: string, question: string, topN: number = 10): Promise<Object[]> {
    this.logger.log(`Starting getProjectInfo with repoUrl=${repoUrl}, question=${question}, topN=${topN}`);
    const localDirectory = './local-directory';

    await this.clearDirectory(localDirectory);

    this.logger.log(`Cloning repository from ${repoUrl} to ${localDirectory}`);
    await simpleGit().clone(repoUrl, localDirectory);

    const files = await this.filredListFiles(localDirectory)


    let docs = []
    for (const fwe of files) {
      this.logger.log(localDirectory + '/' + fwe.file);
      const textLoader = new TextLoader(localDirectory + '/' + fwe.file);
      const doc = await textLoader.load();
      this.logger.log("doc: ");
      this.logger.log(doc);
      docs = docs.concat(doc);
    }
    this.logger.log("docs:");
    this.logger.log(docs);
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 100,
      chunkOverlap: 0,
    });
    const splits = await textSplitter.splitDocuments(docs);
    console.log(splits.length);
    const vectorstore = await MemoryVectorStore.fromDocuments(
      splits,
      embeddings
    );
    this.logger.log(vectorstore);
    this.logger.log('Vector similaritySearch starting');
    const result = await vectorstore.similaritySearch(
      question,
      topN
    );
    this.logger.log(result);
    this.logger.log('Retrieved documents successfully');
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

  async filredListFiles(repo: string): Promise<FileWithExtension[]> {
    try {
      const git = simpleGit();
      await git.cwd(repo);
      const fileList = await git.raw(['ls-tree', '-r', 'HEAD', '--name-only']);
      this.logger.log('Retrieved file list from the repository');
      const files: string[] = fileList.trim().split('\n');
      this.logger.log(`Total files found: ${files.length}`);
      const filteredFiles: FileWithExtension[] = files
        .map(file => {
          const ext = path.extname(file).slice(1);
          return { file, extension: ext };
        })
        .filter(fileObj => (SupportedTextSplitterLanguages as unknown as string[]).includes(fileObj.extension));

      this.logger.log(`Filtered files: ${filteredFiles.length}`);
      this.logger.debug(`Filtered files list: ${filteredFiles.map(f => f.file).join(', ')}`);

      return filteredFiles;
    } catch (error) {
      this.logger.error('Error while filtering files:', error);
      throw error;
    }
  }
}